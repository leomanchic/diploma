"""Actual autocrlf checkout of saved bytes; no synthesis or estimator execution."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def _git(directory: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ['git', *arguments], cwd=directory, capture_output=True, check=True,
        timeout=120,
    ).stdout


def test_autocrlf_checkout_preserves_every_tracked_blob_and_frozen_sha(tmp_path):
    """Use real Git conversion, the working rules, and immutable object bytes.

    Alternates only read existing Git objects; the fixture has its own index,
    object writes and config. It never changes the real worktree or its config.
    Comparing every blob also covers SHA inputs outside the two S8 manifests.
    Nonempty synthetic stderr and mixed newlines cover gaps in the saved logs.
    """
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    _git(checkout, 'init', '--quiet')
    objects = Path(_git(
        REPOSITORY, 'rev-parse', '--path-format=absolute', '--git-path', 'objects'
    ).decode().strip())
    alternates = checkout / '.git/objects/info/alternates'
    # Git treats a CR in this control file as part of the pathname.
    # Do not let Windows text-mode I/O append it.
    alternates.write_bytes((objects.as_posix() + '\n').encode('utf-8'))
    _git(checkout, 'config', 'core.autocrlf', 'false')
    _git(checkout, 'read-tree', _git(REPOSITORY, 'rev-parse', 'HEAD').decode().strip())
    project = checkout / 'uav_acoustic_model'
    project.mkdir()
    attribute_paths = [checkout / '.gitattributes', project / '.gitattributes']
    attribute_paths[0].write_bytes((REPOSITORY / '.gitattributes').read_bytes())
    attribute_paths[1].write_bytes((ROOT / '.gitattributes').read_bytes())
    probes = {
        'checkout_probe.stdout': b'first\nsecond\n',
        'checkout_probe.stderr': b'nonempty diagnostic\nsecond line\n',
        'checkout_probe.unknown_format': b'future SHA input\nsecond line\n',
        'checkout_probe_mixed.stdout': b'CRLF\r\nLF\nlast line\r\n',
        'checkout_probe_mixed.stderr': b'LF\nCRLF\r\n',
    }
    for name, data in probes.items():
        (project / name).write_bytes(data)
    _git(checkout, 'add', '--', '.gitattributes', 'uav_acoustic_model/.gitattributes', *[
        'uav_acoustic_model/' + name for name in probes
    ])
    # A fresh checkout must obtain its attributes from the staged tree.
    for attributes in attribute_paths:
        attributes.unlink()
    for name in probes:
        (project / name).unlink()
    _git(checkout, 'config', 'core.autocrlf', 'true')
    assert _git(checkout, 'config', '--get', 'core.autocrlf').strip() == b'true'
    _git(checkout, 'checkout-index', '--all', '--force')

    object_format = _git(REPOSITORY, 'rev-parse', '--show-object-format').decode().strip()
    entries = _git(checkout, 'ls-files', '--stage', '-z').decode().split('\0')
    for entry in filter(None, entries):
        metadata, name = entry.split('\t', 1)
        mode, expected_blob, stage = metadata.split()
        assert stage == '0' and mode in ('100644', '100755'), name
        data = (checkout / name).read_bytes()
        actual_blob = hashlib.new(
            object_format, b'blob ' + str(len(data)).encode() + b'\0' + data
        ).hexdigest()
        assert actual_blob == expected_blob, f'checkout changed tracked bytes: {name}'
    for name, expected in probes.items():
        assert (project / name).read_bytes() == expected, name

    frozen = json.loads((project / 'HARMONIC_TRACKING_FAILURE_PROTOCOL_MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in frozen['source_files_sha256'].items():
        if name == 'analysis/unseen_manoeuvres_and_sources.py':
            name = frozen['frozen_runner_path']
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((project / 'HARMONIC_TRACKING_FAILURE_PROTOCOL.md').read_bytes()).hexdigest() == frozen['protocol_sha256']
    delivery = json.loads((project / 'results/harmonic_tracking_failure/delivery_manifest.json').read_text(encoding='utf-8'))
    for name, expected in delivery['files_sha256'].items():
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == expected, name
