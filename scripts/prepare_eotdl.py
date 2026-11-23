"""Build the EOTDL upload folder from a clean, tagged commit plus data/; never uploads.

    python scripts/prepare_eotdl.py --tag v1.0.0

Copies every Git-tracked file at the tagged commit and every file under data/ (copies,
not links) into dist/<tag>/spain-biomass-transfer-igarss/, and writes release-manifest.json
with the tag, the commit and the size and SHA-256 hash of each file.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA = ['cci/build_record.json', 'ifn/build_record.json', 'cci/spain/predictions_10m/best_c20', 'cci/spain/example_inputs/best']


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tag', required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Uncommitted changes to tracked files; commit them first.')
    commit = git('rev-parse', 'HEAD')
    if git('rev-parse', f'{args.tag}^{{commit}}') != commit:
        raise SystemExit(f'HEAD is not at tag {args.tag}.')
    readme = (ROOT / 'README.md').read_text()
    name = re.search(r'^name: ([a-z0-9-]{3,45})$', readme.split('---', 2)[1], re.M)[1]
    if 'license: CC-BY-4.0' not in readme.split('---', 2)[1]:
        raise SystemExit('README metadata must declare the CC-BY-4.0 dataset licence.')
    missing = [p for p in REQUIRED_DATA if not (args.data / p).exists()]
    if missing:
        raise SystemExit(f'data/ is incomplete: missing {missing}')

    output = args.output or ROOT / 'dist' / args.tag / name
    output.mkdir(parents=True, exist_ok=False)  # never overwrite a previous snapshot
    tracked = [Path(p) for p in git('ls-files', '-z').split('\0') if p]
    sources = [(ROOT / p, p) for p in tracked]
    sources += [(p, Path('data') / p.relative_to(args.data)) for p in sorted(args.data.rglob('*')) if p.is_file()]
    entries = []
    for src, rel in sources:
        if src.is_symlink() or not src.is_file():
            raise SystemExit(f'Not a regular file: {src}')
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        entries.append({'path': rel.as_posix(), 'bytes': dst.stat().st_size, 'sha256': sha256(dst),
                        'location': 'github+eotdl' if rel.parts[0] != 'data' else 'eotdl'})
    manifest = {'name': name, 'tag': args.tag, 'commit': commit, 'files': entries,
                'note': 'Every payload file except this manifest. "eotdl" files are not in the Git repository.'}
    (output / 'release-manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    total = sum(e['bytes'] for e in entries)
    print(f'Prepared {len(entries)} files ({total / 1e9:.2f} GB) in {output}. Nothing uploaded; see EOTDL.md.')


if __name__ == '__main__':
    main()
