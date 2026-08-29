"""
GGUF in-place metadata patcher.

Rewrites integer metadata values in the GGUF header without touching
tensor data.  Safe because we only overwrite bytes of the same width
at their original file offsets — file size and tensor offsets are
unchanged.
"""

import struct
from typing import Dict, Any, Tuple


# GGUF metadata type ids and their struct formats / byte widths
_TYPE_FMT = {
    0:  ('<B', 1),   # UINT8
    1:  ('<b', 1),   # INT8
    2:  ('<H', 2),   # UINT16
    3:  ('<h', 2),   # INT16
    4:  ('<I', 4),   # UINT32
    5:  ('<i', 4),   # INT32
    6:  ('<f', 4),   # FLOAT32
    7:  ('<?', 1),   # BOOL
    10: ('<Q', 8),   # UINT64
    11: ('<q', 8),   # INT64
    12: ('<d', 8),   # FLOAT64
}


def _skip_value(f, type_id: int) -> None:
    """Advance the file position past one metadata value of the given type."""
    if type_id in _TYPE_FMT:
        _, width = _TYPE_FMT[type_id]
        f.read(width)
    elif type_id == 8:  # STRING
        length = struct.unpack('<Q', f.read(8))[0]
        f.read(length)
    elif type_id == 9:  # ARRAY
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            _skip_value(f, elem_type)
    else:
        raise ValueError(f"Unknown GGUF metadata type id: {type_id}")


def scan_metadata_offsets(filepath: str) -> Dict[str, Tuple[int, int, Any]]:
    """
    Scan the GGUF header and return, for each metadata key:
        (type_id, value_offset, current_value)

    value_offset is the file position of the first byte of the value
    field (i.e. right after the uint32 type tag).
    """
    result: Dict[str, Tuple[int, int, Any]] = {}

    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:
            raise ValueError(f"Not a GGUF file: bad magic {hex(magic)}")
        f.read(4)   # version
        f.read(8)   # tensor count
        meta_count = struct.unpack('<Q', f.read(8))[0]

        for _ in range(meta_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            type_id = struct.unpack('<I', f.read(4))[0]
            value_offset = f.tell()

            # Read current value for reporting
            if type_id in _TYPE_FMT:
                fmt, width = _TYPE_FMT[type_id]
                current = struct.unpack(fmt, f.read(width))[0]
            elif type_id == 8:  # STRING
                slen = struct.unpack('<Q', f.read(8))[0]
                current = f.read(slen).decode('utf-8')
            elif type_id == 9:  # ARRAY — skip & record None
                elem_type = struct.unpack('<I', f.read(4))[0]
                count = struct.unpack('<Q', f.read(8))[0]
                for _ in range(count):
                    _skip_value(f, elem_type)
                current = None
            else:
                raise ValueError(f"Unknown type id {type_id} for key {key!r}")

            result[key] = (type_id, value_offset, current)

    return result


def patch_gguf_metadata_inplace(
    filepath: str,
    patches: Dict[str, Any],
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Patch integer metadata values in a GGUF file in-place.

    Args:
        filepath:  Path to the .gguf file (modified in-place unless dry_run).
        patches:   {key: new_value} — only integer-typed keys are supported.
        dry_run:   If True, report what would change without writing.
        verbose:   Print each change.

    Returns:
        {key: (old_value, new_value)} for every key that was (or would be)
        changed.  Keys not found in the file are silently skipped.
    """
    offsets = scan_metadata_offsets(filepath)
    changed: Dict[str, Tuple[Any, Any]] = {}

    with open(filepath, 'r+b' if not dry_run else 'rb') as f:
        for key, new_val in patches.items():
            if key not in offsets:
                if verbose:
                    print(f"  [SKIP]  {key!r} not found in metadata")
                continue

            type_id, value_offset, old_val = offsets[key]

            if type_id not in _TYPE_FMT:
                if verbose:
                    print(f"  [SKIP]  {key!r} has non-integer type {type_id}, cannot patch")
                continue

            fmt, width = _TYPE_FMT[type_id]
            new_bytes = struct.pack(fmt, int(new_val))

            if old_val == new_val:
                if verbose:
                    print(f"  [OK]    {key} = {old_val} (no change needed)")
                continue

            if verbose:
                action = "(dry run)" if dry_run else ""
                print(f"  [PATCH] {key}: {old_val} → {new_val} {action}")

            if not dry_run:
                f.seek(value_offset)
                f.write(new_bytes)

            changed[key] = (old_val, new_val)

    return changed


def detect_block_count_mismatch(filepath: str) -> Dict[str, Any]:
    """
    Compare GGUF metadata block_count / nextn_predict_layers against
    the actual tensor names in the file.

    Returns a dict with:
        arch_prefix       - e.g. "qwen35"
        meta_block_count  - what the metadata claims
        actual_block_count - max blk.N index + 1
        nextn_key         - the nextn_predict_layers key name
        meta_nextn        - current nextn value
        needs_patch       - True if metadata is inconsistent
        suggested_patches - {key: corrected_value}
    """
    from magicquant.gguf.source import open_model_source
    import re as _re

    src = open_model_source(filepath)
    try:
        tensor_names = src.get_tensor_names()
        metadata = src.get_metadata()
    finally:
        src.close()

    # Find arch prefix
    arch_prefix = None
    for k in metadata:
        if k.endswith(".block_count"):
            arch_prefix = k.rsplit(".", 1)[0]
            break

    if arch_prefix is None:
        return {"needs_patch": False, "reason": "no block_count key found"}

    block_count_key = f"{arch_prefix}.block_count"
    nextn_key = f"{arch_prefix}.nextn_predict_layers"

    meta_block_count = metadata.get(block_count_key)
    meta_nextn = metadata.get(nextn_key, 0)

    # Count actual blocks from tensor names
    blk_indices = set()
    for t in tensor_names:
        m = _re.match(r"blk\.(\d+)\.", t)
        if m:
            blk_indices.add(int(m.group(1)))

    actual_block_count = max(blk_indices) + 1 if blk_indices else None

    if actual_block_count is None or meta_block_count is None:
        return {"needs_patch": False, "reason": "could not determine block counts"}

    needs_patch = meta_block_count != actual_block_count

    suggested: Dict[str, Any] = {}
    if needs_patch:
        suggested[block_count_key] = actual_block_count
        if meta_nextn != 0:
            suggested[nextn_key] = 0

    return {
        "arch_prefix": arch_prefix,
        "block_count_key": block_count_key,
        "meta_block_count": meta_block_count,
        "actual_block_count": actual_block_count,
        "nextn_key": nextn_key,
        "meta_nextn": meta_nextn,
        "needs_patch": needs_patch,
        "suggested_patches": suggested,
    }
