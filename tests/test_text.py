from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

text_module = importlib.import_module("zero_shot_voiceclone.text")
normalize_text = text_module.normalize_text
split_text_into_chunks = text_module.split_text_into_chunks


class TextHelpersTest(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        raw = "  Hello   world.\n\n  Second\tparagraph.  "
        self.assertEqual(normalize_text(raw), "Hello world.\n\nSecond paragraph.")

    def test_split_text_prefers_sentence_boundaries(self) -> None:
        raw = "Sentence one. Sentence two. Sentence three."
        chunks = split_text_into_chunks(raw, max_chars=20)
        self.assertEqual(chunks, ["Sentence one.", "Sentence two.", "Sentence three."])

    def test_split_text_handles_long_runs_without_punctuation(self) -> None:
        raw = "word " * 80
        chunks = split_text_into_chunks(raw, max_chars=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 60 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
