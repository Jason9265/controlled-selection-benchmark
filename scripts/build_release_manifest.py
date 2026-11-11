"""Write deterministic hashes for every distributable release file."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "RELEASE_MANIFEST.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows = []
    prohibited = {".pt", ".pth", ".zip", ".tif", ".tiff"}
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == OUT
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in prohibited:
            raise RuntimeError(f"prohibited distributable artifact: {relative}")
        rows.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path)})
    output = {
        "release": "0.1.0",
        "scope": "Code, metadata, result summaries, receipts, audits, and figures only; no raw data or model weights.",
        "file_count": len(rows),
        "files": rows,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"file_count": len(rows), "manifest": str(OUT)}))


if __name__ == "__main__":
    main()
