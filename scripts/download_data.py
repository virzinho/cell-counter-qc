"""Download and verify BBBC039 archives; preserve the official split lists."""
import hashlib
import json
import logging
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw = ROOT / "data/raw"
    destination = ROOT / "data/bbbc039"
    raw.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    known_path = ROOT / "data/checksums.json"
    known = json.loads(known_path.read_text()) if known_path.exists() else {}
    for filename in ("images.zip", "masks.zip", "metadata.zip"):
        path = raw / filename
        if not path.exists():
            logging.info("Downloading %s", filename)
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve("https://data.broadinstitute.org/bbbc/BBBC039/" + filename, temporary)
            temporary.replace(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if filename in known and checksum != known[filename]:
            raise ValueError(f"Checksum mismatch: {filename}. Remove the archive and try again.")
        checksums[filename] = checksum
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = Path(member.filename)
                if member.is_dir() or name.parts[0] == "__MACOSX" or name.name.startswith("."):
                    continue
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    (ROOT / "data/checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    logging.info("Ready: %s", destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
