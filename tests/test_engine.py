from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

engine_module = importlib.import_module("zero_shot_voiceclone.engine")
domain_module = importlib.import_module("zero_shot_voiceclone.domain")
numpy = importlib.import_module("numpy")
AudioInspection = engine_module.AudioInspection
CLIError = domain_module.CLIError
GenerationSettings = domain_module.GenerationSettings
validate_reference_audio = engine_module.validate_reference_audio


class ReferenceAudioValidationTest(unittest.TestCase):
    def test_long_reference_audio_warns_but_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.wav"
            path.write_bytes(b"placeholder")

            with patch.object(
                engine_module,
                "inspect_audio_file",
                return_value=AudioInspection(45.0, 44100, 1),
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    inspection = validate_reference_audio(path)

        self.assertEqual(inspection.duration_seconds, 45.0)
        self.assertTrue(
            any("longer than 30 seconds" in str(item.message) for item in caught)
        )

    def test_very_short_reference_audio_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.wav"
            path.write_bytes(b"placeholder")

            with patch.object(
                engine_module,
                "inspect_audio_file",
                return_value=AudioInspection(0.5, 44100, 1),
            ):
                with self.assertRaises(CLIError):
                    validate_reference_audio(path)

    def test_engine_synthesizes_through_backend_contract(self) -> None:
        settings = GenerationSettings(
            backend_name="fake",
            model_name="fake-model",
            language="English",
            chunk_max_chars=20,
            silence_ms=10,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []
                self.validated_transcript: str | None = None

            def validate_inputs(self, reference_transcript: str | None) -> None:
                self.validated_transcript = reference_transcript

            def prepare_reference(
                self, reference_audio: Path, reference_transcript: str | None
            ) -> dict[str, object]:
                return {
                    "reference_audio": reference_audio,
                    "reference_transcript": reference_transcript,
                }

            def synthesize_chunk(
                self, text: str, prepared_reference: dict[str, object]
            ) -> tuple[object, int]:
                self.calls.append((text, prepared_reference))
                return numpy.ones(4, dtype=numpy.float32), 1000

        fake_backend = FakeBackend()
        with patch.object(engine_module, "create_backend", return_value=fake_backend):
            engine = engine_module.VoiceCloneEngine(settings)
            audio, sample_rate, chunks = engine.synthesize_document(
                target_text="Sentence one. Sentence two.",
                reference_audio=Path("reference.wav"),
                reference_transcript="Reference transcript",
            )

        self.assertEqual(chunks, ["Sentence one.", "Sentence two."])
        self.assertEqual(sample_rate, 1000)
        self.assertEqual(audio.shape[0], 18)
        self.assertEqual(fake_backend.validated_transcript, "Reference transcript")
        self.assertEqual(len(fake_backend.calls), 2)


if __name__ == "__main__":
    unittest.main()
