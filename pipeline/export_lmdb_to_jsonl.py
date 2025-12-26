"""
Export selected fields (image_path + caption) from an LMDB of pickled dicts to JSON.

Usage:
    python export_lmdb_to_jsonl.py --input_lmdb /path/to/input.lmdb --output_json out.json

Notes:
- Assumes LMDB values are Python dicts pickled with numeric string keys (e.g., "0", "1", ...).
- Image path is derived from common fields (image_path/img_path/path/frame_dir_list), falling back
  to frame_dir_list[0]/000001.jpg if needed.
- Caption key defaults to "caption" but can be overridden via --caption_key; also tries several
  fallback keys commonly used in this repo.
"""

import argparse
import json
import os
import pickle
from typing import Any, Dict, Optional

try:
    import lmdb
except ImportError as e:
    raise SystemExit("lmdb is required. Install with `pip install lmdb`.") from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def extract_image_path(data: Dict[str, Any]) -> Optional[str]:
    """
    Best-effort extraction of an image path from a record.
    """
    for key in ("image_path", "img_path", "path"):
        path = data.get(key)
        if isinstance(path, str) and path:
            return path

    f_list = data.get("frame_dir_list") or data.get("frame_dir")
    if isinstance(f_list, str):
        f_list = [f_list]
    if isinstance(f_list, (list, tuple)) and f_list:
        raw = f_list[0]
        if os.path.isfile(raw):
            return raw
        candidate = os.path.join(raw, "000001.jpg")
        return candidate if os.path.exists(candidate) else raw

    return None


def extract_caption(data: Dict[str, Any], preferred_key: str) -> str:
    """
    Best-effort extraction of a caption.
    """
    if preferred_key and preferred_key in data:
        val = data.get(preferred_key)
        return val if isinstance(val, str) else str(val)

    for key in ("caption", "original_caption", "old_caption", "ground_truth_caption"):
        if key in data:
            val = data.get(key)
            return val if isinstance(val, str) else str(val)
    return ""


def export_lmdb(
    input_lmdb: str,
    output_json: str,
    caption_key: str,
    max_samples: Optional[int],
    skip_meta: bool,
) -> None:
    env = lmdb.open(input_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    count = 0

    with env.begin() as txn, open(output_json, "w", encoding="utf-8") as f_out:
        cursor = txn.cursor()
        f_out.write("[\n")
        first = True
        for key, value in tqdm(cursor, desc="Exporting"):
            if skip_meta and key.endswith(b".meta"):
                continue
            if max_samples is not None and count >= max_samples:
                break
            try:
                data = pickle.loads(value)
            except Exception:
                continue

            image_path = extract_image_path(data)
            caption = extract_caption(data, caption_key)

            if image_path is None and not caption:
                continue

            item = {"image_path": image_path, "caption": caption}
            if not first:
                f_out.write(",\n")
            f_out.write(json.dumps(item, ensure_ascii=False))
            first = False
            count += 1
        f_out.write("\n]\n")

    env.close()
    print(f"Exported {count} samples to {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Export image_path and caption from LMDB to JSONL.")
    parser.add_argument("--input_lmdb", type=str,default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter", help="Path to input LMDB directory.")
    parser.add_argument(
        "--output_json",
        "--output_jsonl",
        dest="output_json",
        type=str,
        default="/ytech_m2v8_hdd/workspace/kling_mm/libozhou/HOI/data/kling_imgcap_100w_origin_cap_yolo_filter.json",
        help="Path to output JSON file (alias --output_jsonl for backward compatibility).",
    )
    parser.add_argument("--caption_key", type=str, default="text", help="Preferred caption key in LMDB records.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of samples to export.")
    parser.add_argument(
        "--skip_meta",
        action="store_true",
        default=True,
        help="Skip keys ending with .meta (useful when LMDB stores paired data/meta entries).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_lmdb):
        raise SystemExit(f"Input LMDB not found: {args.input_lmdb}")

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    export_lmdb(
        args.input_lmdb,
        args.output_json,
        args.caption_key,
        args.max_samples,
        skip_meta=args.skip_meta,
    )


if __name__ == "__main__":
    main()
