"""Build a local Hugging Face model view with a compatible tokenizer.

VERL 0.7 creates several tokenizers directly from ``model.path`` even when
``actor_rollout_ref.model.tokenizer_path`` is set.  A LLaMA-Factory checkpoint
can therefore fail under a newer Transformers release when its serialized
tokenizer metadata uses an older schema.  This module creates a tiny runtime
directory: model and tokenizer files are symlinked from the immutable SFT
checkpoint, while the explicitly selected base tokenizer fills any missing
assets.  This preserves the exact tokenizer saved by SFT.

No model weight is copied and neither source directory is modified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


VIEW_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "rlguard_model_view.json"
_SPECIAL_TOKEN_METADATA = {"tokenizer_config.json", "special_tokens_map.json"}

_TOKENIZER_EXACT_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_PREFIXES = (
    "added_tokens",
    "chat_template",
    "merges",
    "special_tokens_map",
    "tokenization_",
    "tokenizer",
    "vocab",
)


class ModelViewError(RuntimeError):
    """Raised when a safe, deterministic runtime model view cannot be built."""


def _direct_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for candidate in sorted(directory.iterdir(), key=lambda path: path.name):
        if candidate.name == MANIFEST_FILENAME:
            raise ModelViewError(
                f"source directory contains reserved file {MANIFEST_FILENAME}: "
                f"{directory}"
            )
        if candidate.is_file():
            files[candidate.name] = candidate
    return files


def _is_tokenizer_asset(name: str) -> bool:
    lowered = name.lower()
    return lowered in _TOKENIZER_EXACT_NAMES or lowered.startswith(
        _TOKENIZER_PREFIXES
    )


def _merge_legacy_extra_special_tokens(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Translate the pre-4.57 list schema without dropping any token.

    Transformers 4.57 treats ``extra_special_tokens`` as a mapping of named
    model-specific tokens and unconditionally calls ``.keys()``.  Older saved
    tokenizers may contain a list.  A list carries no attribute names, so the
    lossless compatible representation is ``additional_special_tokens``.
    """

    extra = config.get("extra_special_tokens")
    if extra is None or isinstance(extra, dict):
        return config, False
    if not isinstance(extra, list):
        raise ModelViewError(
            "tokenizer metadata has unsupported extra_special_tokens type "
            f"{type(extra).__name__}; expected a mapping or legacy list"
        )

    existing = config.get("additional_special_tokens", [])
    if existing is None:
        existing = []
    if not isinstance(existing, list):
        raise ModelViewError(
            "tokenizer metadata has non-list additional_special_tokens"
        )

    merged = list(existing)
    for token in extra:
        if token not in merged:
            merged.append(token)
    cleaned = dict(config)
    cleaned.pop("extra_special_tokens", None)
    cleaned["additional_special_tokens"] = merged
    return cleaned, True


def _load_clean_tokenizer_metadata(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelViewError(f"cannot read tokenizer config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelViewError(f"tokenizer metadata must be a JSON object: {path}")
    return _merge_legacy_extra_special_tokens(value)


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _view_spec(
    model_path: Path,
    tokenizer_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    model_files = _direct_files(model_path)
    tokenizer_files = {
        name: path
        for name, path in _direct_files(tokenizer_path).items()
        if _is_tokenizer_asset(name)
    }
    if not tokenizer_files:
        raise ModelViewError(
            f"no tokenizer assets were found in explicit tokenizer path: {tokenizer_path}"
        )

    # The SFT checkpoint is authoritative because it records the tokenizer
    # actually used during training. The explicit base tokenizer only fills
    # assets absent from that checkpoint. The launcher points every VERL
    # tokenizer call site at this one view, avoiding split tokenization.
    selected = dict(tokenizer_files)
    selected.update(model_files)
    legacy_schema_repaired = False
    cleaned_metadata_hashes: dict[str, str] = {}
    for name in sorted(_SPECIAL_TOKEN_METADATA):
        source = selected.get(name)
        if source is None:
            continue
        cleaned, repaired = _load_clean_tokenizer_metadata(source)
        legacy_schema_repaired = legacy_schema_repaired or repaired
        canonical = json.dumps(
            cleaned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cleaned_metadata_hashes[name] = hashlib.sha256(canonical).hexdigest()

    spec: dict[str, Any] = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "model_path": str(model_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "legacy_extra_special_tokens_repaired": legacy_schema_repaired,
        "files": {
            name: _file_identity(path)
            for name, path in sorted(selected.items())
        },
    }
    if cleaned_metadata_hashes:
        spec["cleaned_tokenizer_metadata_sha256"] = cleaned_metadata_hashes
    return spec, selected


def _fingerprint(spec: dict[str, Any]) -> str:
    payload = json.dumps(
        spec,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_existing_view(view: Path, spec: dict[str, Any]) -> None:
    if view.is_symlink():
        raise ModelViewError(f"refusing symlink runtime model-view directory: {view}")
    manifest = view / MANIFEST_FILENAME
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            recorded = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelViewError(
            f"runtime model view is incomplete or corrupt: {view}: {exc}"
        ) from exc
    if recorded != spec:
        raise ModelViewError(
            "runtime model view manifest does not match its deterministic name; "
            f"remove only this generated directory and retry: {view}"
        )
    for name, identity in spec["files"].items():
        candidate = view / name
        if not candidate.is_file():
            raise ModelViewError(
                f"runtime model view is missing {name}; remove only {view} and retry"
            )
        if name in _SPECIAL_TOKEN_METADATA:
            cleaned, _repaired = _load_clean_tokenizer_metadata(candidate)
            canonical = json.dumps(
                cleaned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            expected_hash = spec.get("cleaned_tokenizer_metadata_sha256", {}).get(
                name
            )
            if hashlib.sha256(canonical).hexdigest() != expected_hash:
                raise ModelViewError(
                    f"runtime model view metadata changed: {candidate}"
                )
        elif not candidate.is_symlink() or candidate.resolve() != Path(
            identity["path"]
        ):
            raise ModelViewError(
                f"runtime model view link changed: {candidate}"
            )


def build_runtime_model_view(
    model_path: Path,
    tokenizer_path: Path,
    view_root: Path,
) -> tuple[Path, bool]:
    """Return ``(view_path, repaired_legacy_schema)``.

    The final directory name is content-metadata addressed.  Construction is
    staged and renamed atomically, so concurrent launch attempts cannot expose
    a partially populated model directory to Ray workers.
    """

    model_path = model_path.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    view_root = view_root.expanduser().resolve()
    if not model_path.is_dir():
        raise ModelViewError(f"model directory does not exist: {model_path}")
    if not tokenizer_path.is_dir():
        raise ModelViewError(f"tokenizer directory does not exist: {tokenizer_path}")

    spec, selected = _view_spec(model_path, tokenizer_path)
    view_root.mkdir(parents=True, exist_ok=True)
    destination = view_root / f"hf_model_view_{_fingerprint(spec)}"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ModelViewError(f"runtime model view path is not a directory: {destination}")
        _validate_existing_view(destination, spec)
        return destination, bool(spec["legacy_extra_special_tokens_repaired"])

    staging = Path(tempfile.mkdtemp(prefix=".building_hf_model_view_", dir=view_root))
    try:
        for name, source in sorted(selected.items()):
            target = staging / name
            if name in _SPECIAL_TOKEN_METADATA:
                cleaned, _repaired = _load_clean_tokenizer_metadata(source)
                with target.open("w", encoding="utf-8") as handle:
                    json.dump(cleaned, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
            else:
                target.symlink_to(source.resolve())

        stable_spec, _stable_selected = _view_spec(model_path, tokenizer_path)
        if stable_spec != spec:
            raise ModelViewError(
                "model or tokenizer source changed while the runtime view was "
                "being constructed; wait for checkpoint writing to finish and retry"
            )

        # Record only source identities and compatibility decisions; the
        # manifest deliberately contains no API keys or runtime secrets.
        with (staging / MANIFEST_FILENAME).open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir() or destination.is_symlink():
                raise
            # Another launcher completed the same deterministic view first.
            _validate_existing_view(destination, spec)
        return destination, bool(spec["legacy_extra_special_tokens_repaired"])
    finally:
        if staging.exists():
            shutil.rmtree(staging)
