"""Small jsonlines compatibility shim used by LLaMEA logging."""

from __future__ import annotations

import builtins
import json


class _Base:
    def __init__(self, fp):
        self.fp = fp

    def close(self):
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class Writer(_Base):
    def write(self, obj):
        self.fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fp.flush()


class Reader(_Base):
    def __iter__(self):
        for line in self.fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def open(path, mode="r"):
    fp = builtins.open(path, mode, encoding="utf-8")
    if "r" in mode and all(m not in mode for m in "aw+"):
        return Reader(fp)
    return Writer(fp)
