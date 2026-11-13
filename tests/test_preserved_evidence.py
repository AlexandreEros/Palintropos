"""Keep the recovered data, archive receipts and published claims connected."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from tropoi.cli.capsule import read_receipt, verify_capsule

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'docs/validation/run_archive_2026-11-12'
INVENTORY = json.loads((RECORD / 'inventory.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('entry', INVENTORY['capsules'] + INVENTORY['analyses'],
                         ids=lambda entry: entry['id'])
def test_inventory_matches_preserved_bytes_and_full_receipt(entry):
    catalogue = ROOT / entry['catalogue']
    receipt = catalogue / 'SHA256SUMS'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == entry['receipt_sha256']
    archive = json.loads((catalogue / 'archive.json').read_text(encoding='utf-8'))
    sums = read_receipt(receipt)
    assert len(sums) == entry['full_files']
    assert sum(f['size'] for f in archive['files'].values()) == entry['full_bytes']
    assert set(sums) - set(archive['included_files']) == set(entry['omitted_files'])
    assert set(archive['omitted_files']) == set(entry['omitted_files'])
    actual = {p.relative_to(catalogue).as_posix() for p in catalogue.rglob('*') if p.is_file()}
    assert actual == set(archive['included_files']) | {'SHA256SUMS', 'archive.json'}
    assert sum((catalogue / n).stat().st_size for n in actual) == entry['committed_bytes']
    assert verify_capsule(catalogue, catalogue, core=True)['checked'] == entry['core_files']
    assert entry['verification']['files'] == entry['full_files']


@pytest.mark.parametrize('entry', INVENTORY['rescued_pages'], ids=lambda entry: Path(entry['published_path']).name)
def test_rescued_html_bytes_and_git_attributes(entry):
    path = ROOT / entry['published_path']
    assert path.stat().st_size == entry['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    result = subprocess.run(['git','check-attr','text','--',entry['published_path']],
                            cwd=ROOT,capture_output=True,text=True,check=True)
    assert result.stdout.strip().endswith(': text: unset')
    if entry['embedded_input_hashes']:
        page = path.read_text(encoding='utf-8')
        payload = json.loads(re.search(r'<script id="w5-data" type="application/json">(.*?)</script>', page, re.S)[1])
        def hash_records(obj):
            if isinstance(obj, dict):
                if 'hashes' in obj:
                    yield obj['hashes']
                for value in obj.values():
                    yield from hash_records(value)
        assert entry['embedded_input_hashes'] in list(hash_records(payload))
        for name, digest in entry['embedded_input_hashes'].items():
            assert hashlib.sha256((path.parent.parent / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('entry', json.loads((RECORD / 'radiative-provenance.json').read_text()),
                         ids=lambda entry: entry['id'])
def test_radiative_capsule_is_the_published_source(entry):
    capsule = ROOT / 'docs/runs' / entry['id']
    published = ROOT / entry['published_evidence']
    a = (capsule / 'config.json').read_bytes()
    b = (published / 'config.json').read_bytes()
    assert hashlib.sha256(a).hexdigest() == entry['capsule_config_sha256']
    # Tracked legacy evidence outside docs/runs may undergo checkout EOL conversion.
    assert a.replace(b'\r\n', b'\n') == b.replace(b'\r\n', b'\n')
    assert json.loads(a) == json.loads(b)
    for name in entry['byte_identical_outputs']:
        a = (capsule / name).read_bytes()
        b = (published / name).read_bytes()
        if name.endswith('.npz'):
            assert a == b
        else:
            assert a.replace(b'\r\n', b'\n') == b.replace(b'\r\n', b'\n')
    for name in entry['outputs_equal_after_lf']:
        assert (capsule / name).read_bytes().replace(b'\r\n',b'\n') == (published / name).read_bytes().replace(b'\r\n',b'\n')
