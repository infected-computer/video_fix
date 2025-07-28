"""Wrapper module exposing file carving functionality for tests."""
from src.python.file_carver import *  # noqa: F401,F403

class AhoCorasickSearcher:
    """Simplified pattern searcher used in tests."""

    def __init__(self, patterns: list[bytes]):
        self.patterns = patterns
        self.pattern_map = {p: i for i, p in enumerate(patterns)}

    def search(self, data: bytes):
        for pattern in self.patterns:
            start = 0
            while True:
                idx = data.find(pattern, start)
                if idx == -1:
                    break
                yield (idx, pattern)
                start = idx + 1
