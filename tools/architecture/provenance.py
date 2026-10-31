"""Stable identity of the source and generator actually analysed."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path


def working_tree(root: Path, src: str, package: str) -> dict:
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    head = git('rev-parse', 'HEAD')
    def tree(path):
        folder = root / path
        files = sorted(p for p in folder.rglob('*') if p.is_file()
                       and p.suffix in {'.py', '.cu', '.html'}
                       and '__pycache__' not in p.parts)
        # Hash LF-normalized content so the identity is the committed text,
        # independent of checkout line endings (core.autocrlf on Windows).
        hashes = {p.relative_to(root).as_posix():
                  hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  for p in files}
        digest = hashlib.sha256()
        for name, sha in hashes.items():
            digest.update(f'{name}\0{sha}\n'.encode())
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=all', '--', str(path)],
            cwd=root, text=True).splitlines()
        return {'status': 'dirty' if status else 'clean', 'changes': status,
                'sha256': digest.hexdigest(), 'files': hashes}
    return {'base_commit': head, 'source': tree(Path(src) / package),
            'generator': tree(Path('tools/architecture'))}
