"""Refresh archive hashes after an explicitly reviewed maintenance change."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "revision_2026_11"


def main():
    manifest = ROOT / "records/source_manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))
    for row in records:
        row["published_sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_path = ROOT / "SHA256SUMS.json"
    names = json.loads(checksum_path.read_text(encoding="utf-8"))
    values = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    checksum_path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Refreshed {len(values)} hashes; source hashes and numerical files retained.")


if __name__ == "__main__":
    main()
