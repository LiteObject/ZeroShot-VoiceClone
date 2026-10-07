from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

backends_module = importlib.import_module("zero_shot_voiceclone.backends")
domain_module = importlib.import_module("zero_shot_voiceclone.domain")
numpy = importlib.import_module("numpy")
qwen_module = importlib.import_module("zero_shot_voiceclone.backends.qwen")
xtts_module = importlib.import_module("zero_shot_voiceclone.backends.xtts")

GenerationSettings = domain_module.GenerationSettings
CLIError = domain_module.CLIError
QwenVoiceCloneBackend = qwen_module.QwenVoiceCloneBackend
XTTSVoiceCloneBackend = xtts_module.XTTSVoiceCloneBackend
get_backend_names = backends_module.get_backend_names


class BackendRegistryTest(unittest.TestCase):
    def test_registry_lists_xtts_backend(self) -> None:
        self.assertIn("xtts", get_backend_names())


class QwenBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = QwenVoiceCloneBackend(
            GenerationSettings(
                backend_name="qwen",
                model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                language="English",
                chunk_max_chars=240,
                silence_ms=250,
                backend_options={},
            )
        )

    def test_qwen_limits_generation_by_default(self) -> None:
        prepared_reference = object()
        with patch.object(self.backend, "_load_model") as load_model:
            model = load_model.return_value
            model.generate_voice_clone.return_value = ([[0.1, -0.1]], 24000)
            waveform, sample_rate = self.backend.synthesize_chunk(
                "Hello world.", prepared_reference
            )

        model.generate_voice_clone.assert_called_once_with(
            text="Hello world.",
            language="English",
            voice_clone_prompt=prepared_reference,
            max_new_tokens=2048,
        )
        self.assertEqual(waveform, [0.1, -0.1])
        self.assertEqual(sample_rate, 24000)

    def test_qwen_forwards_custom_generation_limit(self) -> None:
        self.backend.settings.backend_options["max_new_tokens"] = 512
        with patch.object(self.backend, "_load_model") as load_model:
            model = load_model.return_value
            model.generate_voice_clone.return_value = ([[0.1]], 24000)
            self.backend.synthesize_chunk("Hello world.", object())

        self.assertEqual(model.generate_voice_clone.call_args.kwargs["max_new_tokens"], 512)

    def test_qwen_rejects_nonpositive_generation_limit(self) -> None:
        for limit in (0, -1):
            with self.subTest(limit=limit):
                self.backend.settings.backend_options["max_new_tokens"] = limit
                with self.assertRaisesRegex(CLIError, "must be greater than zero"):
                    self.backend.validate_inputs("Reference speech.")


class XTTSBackendTest(unittest.TestCase):
    def test_xtts_rejects_auto_language(self) -> None:
        backend = XTTSVoiceCloneBackend(
            GenerationSettings(
                backend_name="xtts",
                model_name="coqui/XTTS-v2",
                language="Auto",
                chunk_max_chars=240,
                silence_ms=250,
                backend_options={"device": "cpu"},
            )
        )

        with self.assertRaises(CLIError):
            backend.validate_inputs(None)

    def test_xtts_ignores_reference_transcript(self) -> None:
        backend = XTTSVoiceCloneBackend(
            GenerationSettings(
                backend_name="xtts",
                model_name="coqui/XTTS-v2",
                language="en",
                chunk_max_chars=240,
                silence_ms=250,
                backend_options={"device": "cpu"},
            )
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend.validate_inputs("This transcript is ignored.")

        self.assertTrue(
            any("ignores --reference-text-file" in str(item.message) for item in caught)
        )

    def test_xtts_synthesizes_with_normalized_language(self) -> None:
        backend = XTTSVoiceCloneBackend(
            GenerationSettings(
                backend_name="xtts",
                model_name="coqui/XTTS-v2",
                language="English",
                chunk_max_chars=240,
                silence_ms=250,
                backend_options={"device": "cpu", "split_sentences": False},
            )
        )

        class FakeSynthesizer:
            output_sample_rate = 24000

        class FakeTTS:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.synthesizer = FakeSynthesizer()

            def tts(self, **kwargs):
                self.calls.append(kwargs)
                return [0.1, -0.1, 0.2]

        fake_tts = FakeTTS()
        prepared = backend.prepare_reference(Path("speaker.wav"), None)

        with patch.object(backend, "_load_model", return_value=fake_tts):
            waveform, sample_rate = backend.synthesize_chunk("Hello world", prepared)

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(fake_tts.calls[0]["language"], "en")
        self.assertEqual(fake_tts.calls[0]["speaker_wav"], ["speaker.wav"])
        self.assertFalse(fake_tts.calls[0]["split_sentences"])
        self.assertEqual(waveform.dtype, numpy.float32)
        self.assertEqual(waveform.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
