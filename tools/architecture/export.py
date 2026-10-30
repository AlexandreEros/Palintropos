"""Export a generated snapshot with metrics, provenance and Mermaid sources.

Usage: python -m tools.architecture.export --out /workspace/downloads/architecture
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def export(snapshot: Path, out: Path) -> tuple[Path, Path]:
    files = sorted(p for p in snapshot.rglob('*') if p.is_file())
    metrics = json.loads((snapshot / 'metrics.json').read_text())
    if not metrics['source'].get('commit') and not metrics['source'].get('working_tree'):
        raise ValueError('Generate a snapshot with source provenance before exporting')
    out.parent.mkdir(parents=True, exist_ok=True)
    md = out.with_suffix('.md')
    parts = [(snapshot / 'README.md').read_text(), '\n# Measured results\n',
             (snapshot / 'metrics.md').read_text(),
             '\n# Machine-readable results and provenance\n\n```json\n' +
             (snapshot / 'metrics.json').read_text() + '\n```\n', '\n# Diagrams\n']
    for p in files:
        if p.suffix == '.mmd':
            parts.append(f'\n## {p.relative_to(snapshot).as_posix()}\n\n```mermaid\n{p.read_text()}\n```\n')
    review = snapshot.parent / 'review'
    context = sorted(p for p in review.rglob('*') if p.is_file()) if review.is_dir() else []
    for p in context:
        parts.append('\n# Review context: ' + p.name + '\n\n' +
                     ('```json\n' + p.read_text() + '\n```' if p.suffix == '.json' else p.read_text()))
    md.write_text('\n'.join(parts))
    archive = out.with_suffix('.zip')
    contents = {p.relative_to(snapshot).as_posix(): p.read_bytes() for p in files}
    contents.update({'review/' + p.relative_to(review).as_posix(): p.read_bytes() for p in context})
    contents['combined.md'] = md.read_bytes()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(contents.items()): z.writestr(name, data)
        z.writestr('SHA256SUMS', ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                                       for name, data in sorted(contents.items())))
    return md, archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=Path('docs/architecture/current'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for path in export(args.snapshot, args.out): print(path)


if __name__ == '__main__': main()
