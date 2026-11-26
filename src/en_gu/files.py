"""Atomic writes for release records and generated data files."""

import os
import tempfile


def atomic_write(path, content, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(content)
    try:
        os.chmod(temp.name, mode)
        os.replace(temp.name, path)
    finally:
        if os.path.exists(temp.name):
            os.unlink(temp.name)
