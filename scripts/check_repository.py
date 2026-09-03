"""Fail if the GitHub upload candidate contains data or unsafe local paths."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".sqlite",
    ".db",
    ".pptx",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
    ".cff",
    ".gitignore",
    ".gitattributes",
}
MAX_PUBLIC_FILE_BYTES = 20 * 1024 * 1024


def upload_candidates() -> list[Path]:
    """Return tracked and unignored untracked files from this repository."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    findings: list[str] = []
    try:
        repository_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Repository check requires an initialized Git repository.", file=sys.stderr)
        return 2

    if repository_root != ROOT.resolve():
        print(
            f"Run inside the dedicated repository. Git root is {repository_root}, expected {ROOT.resolve()}.",
            file=sys.stderr,
        )
        return 2

    for path in upload_candidates():
        relative = path.relative_to(ROOT)
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden data/model file: {relative}")
        if relative.parts and relative.parts[0] == "data" and relative != Path("data/README.md"):
            findings.append(f"only data/README.md may be uploaded: {relative}")
        if path.name.startswith("~$"):
            findings.append(f"temporary Office file: {relative}")
        if path.is_file() and path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            findings.append(f"file exceeds 20 MiB review threshold: {relative}")
        if suffix in TEXT_SUFFIXES or path.name in {"README", "LICENSE", "Makefile"}:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                findings.append(f"text file is not UTF-8: {relative}")
                continue
            # Build the signatures in pieces so this checker does not flag itself.
            for marker in ("/" + "Users/", "C:" + "\\Users\\"):
                if marker in content:
                    findings.append(f"hard-coded local path in {relative}: {marker}")

    if findings:
        print("Repository hygiene check failed:", file=sys.stderr)
        for finding in sorted(set(findings)):
            print(f"- {finding}", file=sys.stderr)
        return 1

    print(f"Repository hygiene check passed for {len(upload_candidates())} upload candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
