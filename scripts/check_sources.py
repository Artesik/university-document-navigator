"""Read-only integrity check of the summer snapshot and supplied materials."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def verify():
    snapshot = ROOT / 'legacy-source/synergy-ai-assistant'
    manifest = json.loads((ROOT / 'docs/legacy-source-manifest.json').read_text('utf-8-sig'))
    expected = {row['path'].replace('\\', '/'): row['sha256'].lower() for row in manifest}
    if snapshot.is_dir():
        actual = {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in snapshot.rglob('*') if p.is_file()}
    elif snapshot.with_suffix('.zip').is_file():
        with ZipFile(snapshot.with_suffix('.zip')) as archive:
            names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
            prefix = 'synergy-ai-assistant/'
            if len(names) != len(set(names)) or any(not name.startswith(prefix) for name in names):
                raise SystemExit('Unexpected entries in summer snapshot archive')
            actual = {name.removeprefix(prefix): hashlib.sha256(archive.read(name)).hexdigest()
                      for name in names}
    else:
        raise SystemExit('Summer snapshot directory or ZIP not found')
    if actual != expected:
        raise SystemExit('Snapshot mismatch: ' + str(sorted(k for k in actual.keys() | expected.keys()
                                                           if actual.get(k) != expected.get(k))))
    print(f'Snapshot verified: {len(actual)} files, SHA-256 match.')
    if not (ROOT / 'materials').exists():
        print('Materials not distributed in this source package; snapshot verification complete.')
        return
    materials = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT / 'materials').rglob('*') if p.is_file()}
    saved = ROOT / 'docs/materials-sha256.json'
    if saved.exists():
        assert json.loads(saved.read_text('utf-8')) == materials, 'Materials changed'
    else:
        saved.write_text(json.dumps(materials, indent=2), encoding='utf-8')
    print(f'Materials verified: {len(materials)} files.')


if __name__ == '__main__':
    verify()
