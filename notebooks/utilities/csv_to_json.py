# csv_to_jsonl.py
import csv
import json
import sys
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Simple helper to coerce a JSON-like string into a proper JSON string value.
def normalize_json_field(v: str):
    """Try to parse `v` as JSON, then as Python literal, then attempt to
    extract a brace-delimited substring. Returns a Python object (list/dict)
    or None if parsing failed.
    """
    if not v or not isinstance(v, str):
        return None

    # 1) Try proper JSON
    try:
        return json.loads(v)
    except Exception:
        pass

    # 2) Try Python literal (single quotes, etc.)
    try:
        return ast.literal_eval(v)
    except Exception:
        pass

    # 3) Best-effort: extract the substring between the first '[' and the last ']' or
    # between first '{' and last '}' (handle lists of dicts or single dicts)
    for (start_char, end_char) in (('[', ']'), ('{', '}')):
        start = v.find(start_char)
        end = v.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            sub = v[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                try:
                    return ast.literal_eval(sub)
                except Exception:
                    logging.debug('Failed to parse extracted substring: %r', sub)

    # 4) Give up and return the original string (consumer can decide what to do)
    logging.warning('Could not parse field value, leaving as raw string (truncated): %r', v[:200])
    return v


def main(argv):
    if len(argv) < 3:
        print('Usage: python csv_to_json.py <in.csv> <out.jsonl>')
        return 2

    inp = argv[1]
    out = argv[2]

    with open(inp, newline='', encoding='utf-8') as f_in, open(out, 'w', encoding='utf-8') as f_out:
        r = csv.DictReader(f_in)
        for row in r:
            # Ensure nested fields are parsed into Python objects (lists/dicts)
            for k in ("landmarks", "bounding_box"):
                v = row.get(k)
                if v and isinstance(v, str):
                    parsed = normalize_json_field(v)
                    # If parsing returned a Python object, assign it; otherwise keep raw string
                    row[k] = parsed
            # Now dump the whole record — nested objects will be serialized as JSON
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print('done.')


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))