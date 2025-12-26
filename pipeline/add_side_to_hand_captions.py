"""
Read a JSON file, find captions containing the bare word "hand" without a nearby
"left"/"right" before it, and inject a random "left" or "right" before that word.

The detection is intentionally strict to avoid false positives:
- Only the exact word "hand" (case-insensitive) is considered (not "hands"/"handed"/"handbag").
- If any of the last 3 words before "hand" is left/right, no change is made.

Usage:
  python add_side_to_hand_captions.py --input_json in.json --output_json out.json
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple


HAND_PATTERN = re.compile(r"\bhand\b", re.IGNORECASE)


def load_json_records(path: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load JSON; accept list, or dict containing list under results/data/items.
    Returns (records, wrapper_meta) so we can reconstruct the original shape.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, None

    if isinstance(data, dict):
        for key in ("results", "data", "items"):
            if key in data and isinstance(data[key], list):
                meta = {k: v for k, v in data.items() if k != key}
                return data[key], {"key": key, "meta": meta}

    raise ValueError("Input JSON must be a list or have a list under results/data/items.")


def wrap_json_records(records: List[Dict[str, Any]], wrapper: Optional[Dict[str, Any]]) -> Any:
    """Rebuild original structure if input was wrapped."""
    if wrapper is None:
        return records
    result = dict(wrapper.get("meta", {}))
    result[wrapper["key"]] = records
    return result


def has_side_in_prefix(prefix: str) -> bool:
    """
    Check last up-to-3 words in prefix for 'left' or 'right'.
    """
    words = re.findall(r"[A-Za-z]+", prefix.lower())
    return any(w in ("left", "right") for w in words[-3:])


def insert_side(caption: str) -> Tuple[str, bool]:
    """
    Insert 'left' or 'right' before bare 'hand' when no side appears nearby.
    Returns (new_caption, changed_flag).
    """
    changed = False

    def repl(match: re.Match) -> str:
        nonlocal changed
        start = match.start()
        prefix = caption[:start]
        if has_side_in_prefix(prefix):
            return match.group(0)
        side = random.choice(["left", "right"])
        changed = True
        return f"{side} {match.group(0)}"

    new_caption = HAND_PATTERN.sub(repl, caption)
    return new_caption, changed


def process_records(
    records: List[Dict[str, Any]],
    caption_key: str,
) -> int:
    """
    Modify captions in-place; returns number of captions changed.
    """
    changed = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        caption = rec.get(caption_key)
        if not isinstance(caption, str):
            continue
        new_caption, did_change = insert_side(caption)
        if did_change:
            rec[caption_key] = new_caption
            changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Insert random left/right before bare 'hand' in captions."
    )
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file (list or wrapped list).")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save modified JSON.")
    parser.add_argument("--caption_key", type=str, default="caption", help="JSON key for caption text.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    records, wrapper = load_json_records(args.input_json)
    changed = process_records(records, caption_key=args.caption_key)

    out_obj = wrap_json_records(records, wrapper)
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(records)} records, changed {changed} captions.")


if __name__ == "__main__":
    main()
