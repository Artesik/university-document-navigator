"""Verify distributed file bytes using the standard library; never modifies files."""
import hashlib
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    manifest = root / 'FILES.sha256'
    if not manifest.is_file():
        raise SystemExit('FILES.sha256 not found. Run this script from the distributed source package.')
    errors, seen = [], set()
    for line in manifest.read_text('utf-8').splitlines():
        checksum, separator, name = line.partition('  ')
        path = PurePosixPath(name)
        if (not separator or not re.fullmatch(r'[0-9a-f]{64}', checksum)
                or not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or name in seen):
            raise SystemExit('Invalid entry in FILES.sha256')
        seen.add(name)
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
            raise SystemExit('Unexpected file path: ' + name)
        if not target.is_file():
            errors.append('Missing: ' + name)
        elif hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            errors.append('Changed: ' + name)
    if not seen:
        raise SystemExit('Empty FILES.sha256')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Package verified: {len(seen)} file hashes match. FILES.sha256 itself is not included.')


if __name__ == '__main__':
    verify()
