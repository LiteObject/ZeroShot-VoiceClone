from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

cli_module = importlib.import_module("zero_shot_voiceclone.cli")
DEFAULT_BACKEND = cli_module.DEFAULT_BACKEND
DEFAULT_MODEL = cli_module.DEFAULT_MODEL
build_backend_options = cli_module.build_backend_options
build_parser = cli_module.build_parser


class CliParserTest(unittest.TestCase):
    def test_synth_parser_uses_expected_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "synth",
                "--reference-audio",
                "sample.wav",
                "--reference-text-file",
                "sample.txt",
                "--target-text-file",
                "script.txt",
                "--output",
                "out.wav",
                "--confirm-rights-to-voice",
            ]
        )
        self.assertEqual(args.backend, DEFAULT_BACKEND)
        self.assertIsNone(args.model)
        self.assertEqual(args.language, "Auto")
        self.assertEqual(args.chunk_max_chars, 240)
        self.assertEqual(args.silence_ms, 250)
        self.assertFalse(args.qwen_x_vector_only)
        self.assertTrue(args.confirm_rights_to_voice)

    def test_xtts_specific_flags_are_parsed(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "synth",
                "--backend",
                "xtts",
                "--reference-audio",
                "sample.wav",
                "--target-text-file",
                "script.txt",
                "--output",
                "out.wav",
                "--language",
                "en",
                "--xtts-device",
                "cpu",
                "--xtts-split-sentences",
                "--confirm-rights-to-voice",
            ]
        )
        self.assertEqual(args.backend, "xtts")
        self.assertEqual(args.xtts_device, "cpu")
        self.assertTrue(args.xtts_split_sentences)

    def test_backend_option_builder_returns_xtts_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "synth",
                "--backend",
                "xtts",
                "--reference-audio",
                "sample.wav",
                "--target-text-file",
                "script.txt",
                "--output",
                "out.wav",
                "--language",
                "en",
                "--xtts-device",
                "cpu",
                "--no-xtts-split-sentences",
                "--confirm-rights-to-voice",
            ]
        )
        self.assertEqual(
            build_backend_options(args),
            {"device": "cpu", "split_sentences": False},
        )


if __name__ == "__main__":
    unittest.main()
