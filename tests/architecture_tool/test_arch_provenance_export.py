"""Exports identify dirty source and include measured results, not just definitions."""
import json
import re
import subprocess
import zipfile

import pytest

from tools.architecture.provenance import working_tree
from tools.architecture.export import export


def test_source_identity_tracks_uncommitted_bytes_and_untracked_files(tmp_path):
    def git(*args):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True)
    git('init')
    folder = tmp_path / 'src' / 'pkg'
    folder.mkdir(parents=True)
    module = folder / '__init__.py'
    module.write_text('value = 1\n')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
    first = working_tree(tmp_path, 'src', 'pkg')
    assert first['source']['status'] == 'clean'
    module.write_text('value = 2\n')
    (folder / 'new.py').write_text('value = 3\n')
    second = working_tree(tmp_path, 'src', 'pkg')
    assert second['base_commit'] == first['base_commit']
    assert second['source']['status'] == 'dirty'
    assert second['source']['sha256'] != first['source']['sha256']
    assert 'src/pkg/new.py' in second['source']['files']
    assert second == working_tree(tmp_path, 'src', 'pkg')


def test_export_includes_results_provenance_and_original_mermaid(tmp_path):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'README.md').write_text('# Provenance\n')
    (snapshot / 'metrics.md').write_text('| Nodes | 12 |\n')
    (snapshot / 'metrics.json').write_text(json.dumps({'source': {'commit': 'abc'}}))
    code = 'flowchart LR\n  A --> B\n'
    (snapshot / 'modules.mmd').write_bytes(code.encode())
    md, archive = export(snapshot, tmp_path / 'download')
    assert '| Nodes | 12 |' in md.read_text()
    assert '"commit": "abc"' in md.read_text()
    with zipfile.ZipFile(archive) as z:
        assert z.read('modules.mmd').decode() == code
        assert {'metrics.json', 'metrics.md', 'SHA256SUMS', 'combined.md'} <= set(z.namelist())


def test_check_survives_commits_but_rejects_input_and_output_changes(tmp_path, monkeypatch):
    from tools.architecture import generate

    def git(*args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Fixture', '-c',
             'user.email=fixture@example.invalid', *args],
            cwd=tmp_path, text=True).strip()

    git('init')
    module = tmp_path / 'src' / 'pkg' / '__init__.py'
    module.parent.mkdir(parents=True)
    module.write_text('def entry():\n    return 1\n')
    generator = tmp_path / 'tools' / 'architecture' / 'generate.py'
    generator.parent.mkdir(parents=True)
    generator.write_text('# generator identity fixture\n')
    git('add', '.')
    git('commit', '-m', 'source')
    source_commit = git('rev-parse', 'HEAD')
    monkeypatch.setattr(generate, 'REPO_ROOT', tmp_path)
    args = ['--package', 'pkg', '--out', 'snapshot',
            '--call-root', 'entry=pkg.entry']
    assert generate.main([*args, '--source-commit', source_commit]) == 0
    git('add', '.')
    git('commit', '-m', 'record snapshot')
    assert generate.main([*args, '--check']) == 0
    # A later unrelated commit must also leave the snapshot current.
    (tmp_path / 'notes.txt').write_text('unrelated\n')
    git('add', '.')
    git('commit', '-m', 'notes')
    assert generate.main([*args, '--check']) == 0

    # Even a comment-only input change must be detected by content identity.
    for path in (module, generator):
        original = path.read_text()
        path.write_text(original + '# changed\n')
        assert generate.main([*args, '--check']) == 1
        path.write_text(original)
    diagram = tmp_path / 'snapshot' / 'modules.mmd'
    original = diagram.read_text()
    diagram.write_text('tampered\n')
    assert generate.main([*args, '--check']) == 1
    diagram.write_text(original)

    # Generating while dirty then committing both source and output is valid.
    module.write_text('def entry():\n    return 2\n')
    assert generate.main(args) == 0
    git('add', '.')
    git('commit', '-m', 'source and snapshot together')
    assert generate.main([*args, '--check']) == 0


def test_source_row_names_working_tree_state():
    from tools.architecture.report import _source_row

    def source(status):
        return {'commit': None, 'working_tree': {
            'base_commit': 'abc', 'source': {'status': status}}}

    assert _source_row({'commit': 'def'}) == '| Source commit | `def` |'
    assert _source_row(source('clean')) == (
        '| Source base commit | `abc` (clean working tree) |')
    assert 'uncommitted' in _source_row(source('dirty'))
    assert 'not recorded' in _source_row({'commit': None})


def test_content_hashes_ignore_checkout_line_endings(tmp_path):
    from tools.architecture.provenance import working_tree

    def snapshot(text: bytes):
        repo = tmp_path / ('crlf' if b'\r' in text else 'lf')
        module = repo / 'src' / 'pkg' / '__init__.py'
        module.parent.mkdir(parents=True)
        module.write_bytes(text)
        (repo / 'tools' / 'architecture').mkdir(parents=True)
        for args in (['init'], ['-c', 'core.autocrlf=false', 'add', '.'],
                     ['-c', 'user.name=Fixture', '-c',
                      'user.email=fixture@example.invalid', 'commit', '-m', 'x']):
            subprocess.check_output(['git', *args], cwd=repo)
        return working_tree(repo, 'src', 'pkg')['source']

    lf, crlf = snapshot(b'a = 1\nb = 2\n'), snapshot(b'a = 1\r\nb = 2\r\n')
    assert lf['sha256'] == crlf['sha256']
    assert lf['files'] == crlf['files']


@pytest.mark.parametrize('spelling', ['HEAD', 'short', 'full'])
def test_source_commit_spelling_does_not_change_the_recorded_command(
        tmp_path, monkeypatch, spelling):
    from tools.architecture import generate

    def git(*args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Fixture', '-c',
             'user.email=fixture@example.invalid', *args],
            cwd=tmp_path, text=True).strip()

    git('init')
    module = tmp_path / 'src' / 'pkg' / '__init__.py'
    module.parent.mkdir(parents=True)
    module.write_text('def entry():\n    return 1\n')
    generator = tmp_path / 'tools' / 'architecture' / 'generate.py'
    generator.parent.mkdir(parents=True)
    generator.write_text('# generator identity fixture\n')
    git('add', '.')
    git('commit', '-m', 'source')
    full = git('rev-parse', 'HEAD')
    rev = {'HEAD': 'HEAD', 'short': full[:7], 'full': full}[spelling]
    monkeypatch.setattr(generate, 'REPO_ROOT', tmp_path)
    args = ['--package', 'pkg', '--out', 'snapshot',
            '--call-root', 'entry=pkg.entry']

    assert generate.main([*args, '--source-commit', rev]) == 0
    readme = (tmp_path / 'snapshot' / 'README.md').read_text()
    recorded = re.findall(r'--source-commit (\S+?)[`\s]', readme)
    assert recorded and set(recorded) == {full}
    for check_rev in ([], ['--source-commit', rev],
                      ['--source-commit', full]):
        assert generate.main([*args, '--check', *check_rev]) == 0
