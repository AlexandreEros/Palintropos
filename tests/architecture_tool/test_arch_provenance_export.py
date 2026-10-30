"""Exports identify dirty source and include measured results, not just definitions."""
import json
import subprocess
import zipfile

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
    (snapshot / 'modules.mmd').write_text(code)
    md, archive = export(snapshot, tmp_path / 'download')
    assert '| Nodes | 12 |' in md.read_text()
    assert '"commit": "abc"' in md.read_text()
    with zipfile.ZipFile(archive) as z:
        assert z.read('modules.mmd').decode() == code
        assert {'metrics.json', 'metrics.md', 'SHA256SUMS', 'combined.md'} <= set(z.namelist())
