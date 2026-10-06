from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "transcribe_audio.py"

_spec = importlib.util.spec_from_file_location("transcribe_audio", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
transcribe_audio = importlib.util.module_from_spec(_spec)
sys.modules["transcribe_audio"] = transcribe_audio
_spec.loader.exec_module(transcribe_audio)

Segment = transcribe_audio.Segment
TranscribeError = transcribe_audio.TranscribeError


class FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    def __init__(self, language: str | None = "en", probability: float | None = 0.98):
        self.language = language
        self.language_probability = probability


def make_model_class(
    segments: list[FakeSegment],
    *,
    fail_load_devices: tuple[str, ...] = (),
    fail_transcribe_devices: tuple[str, ...] = (),
) -> tuple[type, list[tuple[str, str, str]], list[dict[str, object]]]:
    """Build a stand-in WhisperModel class that records how it was used."""
    loads: list[tuple[str, str, str]] = []
    calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            if device in fail_load_devices:
                raise RuntimeError(f"{device} unavailable")
            self.device = device
            loads.append((model_name, device, compute_type))

        def transcribe(
            self,
            audio: str,
            language: str | None = None,
            vad_filter: bool = True,
            beam_size: int = 5,
        ) -> tuple[object, FakeInfo]:
            calls.append(
                {
                    "audio": audio,
                    "language": language,
                    "vad_filter": vad_filter,
                    "beam_size": beam_size,
                }
            )
            if self.device in fail_transcribe_devices:
                raise RuntimeError(f"{self.device} inference unavailable")
            return iter(segments), FakeInfo()

    return FakeModel, loads, calls


class TranscriptTextTest(unittest.TestCase):
    def test_segments_to_text_normalizes_whitespace(self) -> None:
        text = transcribe_audio.segments_to_text(
            [
                Segment(0.0, 1.0, " Hello  there\nworld "),
                Segment(1.0, 2.0, " Again. "),
            ]
        )
        self.assertEqual(text, "Hello there world Again.")

    def test_segments_to_text_handles_empty(self) -> None:
        self.assertEqual(transcribe_audio.segments_to_text([]), "")

    def test_format_timestamp(self) -> None:
        self.assertEqual(transcribe_audio.format_timestamp(0), "00:00")
        self.assertEqual(transcribe_audio.format_timestamp(62.9), "01:02")
        self.assertEqual(transcribe_audio.format_timestamp(3725), "1:02:05")

    def test_format_segments_includes_range_and_text(self) -> None:
        rendered = transcribe_audio.format_segments([Segment(0.0, 4.6, "Hello there.")])
        self.assertIn("[00:00 - 00:04]", rendered)
        self.assertIn("Hello there.", rendered)


class TranscribeFileTest(unittest.TestCase):
    def _transcribe(self, model_class: type, **overrides: object) -> Any:
        options: dict[str, object] = {
            "model_name": "large-v3",
            "device": "auto",
            "compute_type": None,
            "language": None,
            "vad": True,
        }
        options.update(overrides)
        return transcribe_audio.transcribe_file(
            Path("clip.wav"), model_class=model_class, **options
        )

    def test_auto_prefers_cuda_with_float16(self) -> None:
        model_class, loads, _ = make_model_class([FakeSegment(0.0, 1.0, "Hi.")])
        outcome = self._transcribe(model_class)

        self.assertEqual(loads, [("large-v3", "cuda", "float16")])
        self.assertTrue(outcome.device_label.startswith("cuda"))
        self.assertEqual(outcome.fallback_notes, [])
        self.assertEqual(outcome.language, "en")

    def test_auto_falls_back_when_cuda_load_fails(self) -> None:
        model_class, loads, _ = make_model_class(
            [FakeSegment(0.0, 1.0, "Hi.")], fail_load_devices=("cuda",)
        )
        outcome = self._transcribe(model_class)

        self.assertEqual(loads, [("large-v3", "cpu", "int8")])
        self.assertEqual(outcome.device_label, "cpu, int8")
        self.assertTrue(any("cuda" in note for note in outcome.fallback_notes))

    def test_auto_falls_back_when_cuda_inference_fails(self) -> None:
        model_class, loads, calls = make_model_class(
            [FakeSegment(0.0, 1.0, "Hi.")], fail_transcribe_devices=("cuda",)
        )
        outcome = self._transcribe(model_class)

        self.assertEqual([entry[1] for entry in loads], ["cuda", "cpu"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome.device_label, "cpu, int8")
        self.assertTrue(any("cuda" in note for note in outcome.fallback_notes))

    def test_explicit_device_does_not_fall_back(self) -> None:
        model_class, _, _ = make_model_class(
            [], fail_load_devices=("cuda",)
        )
        with self.assertRaises(TranscribeError):
            self._transcribe(model_class, device="cuda")

    def test_compute_type_override_is_used(self) -> None:
        model_class, loads, _ = make_model_class([FakeSegment(0.0, 1.0, "Hi.")])
        self._transcribe(model_class, model_name="small", device="cpu", compute_type="float32")

        self.assertEqual(loads, [("small", "cpu", "float32")])


class CliRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)
        self.input = self.tmp_path / "reference.wav"
        self.input.write_bytes(b"stub")

    def _run(
        self, argv: list[str], model_class: type | None = None
    ) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = transcribe_audio.main(argv, model_class=model_class)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_writes_default_transcript_file(self) -> None:
        model_class, _, calls = make_model_class(
            [FakeSegment(0.0, 2.5, " On a bright autumn morning.")]
        )
        exit_code, stdout, _ = self._run(
            [str(self.input), "--language", "en"], model_class=model_class
        )

        self.assertEqual(exit_code, 0)
        transcript = self.tmp_path / "reference.txt"
        self.assertTrue(transcript.exists())
        self.assertEqual(
            transcript.read_text(encoding="utf-8"),
            "On a bright autumn morning.\n",
        )
        self.assertEqual(calls[-1]["language"], "en")
        self.assertTrue(calls[-1]["vad_filter"])
        self.assertIn("Transcript:", stdout)

    def test_auto_language_passes_none_to_model(self) -> None:
        model_class, _, calls = make_model_class([FakeSegment(0.0, 1.0, "Hi.")])
        exit_code, _, _ = self._run([str(self.input)], model_class=model_class)

        self.assertEqual(exit_code, 0)
        self.assertIsNone(calls[-1]["language"])

    def test_no_vad_flag_disables_vad(self) -> None:
        model_class, _, calls = make_model_class([FakeSegment(0.0, 1.0, "Hi.")])
        exit_code, _, _ = self._run(
            [str(self.input), "--no-vad"], model_class=model_class
        )

        self.assertEqual(exit_code, 0)
        self.assertFalse(calls[-1]["vad_filter"])

    def test_no_speech_reports_error_without_writing_output(self) -> None:
        model_class, _, _ = make_model_class([])
        exit_code, _, stderr = self._run([str(self.input)], model_class=model_class)

        self.assertEqual(exit_code, 2)
        self.assertIn("No speech was detected", stderr)
        self.assertFalse((self.tmp_path / "reference.txt").exists())

    def test_refuses_overwrite_without_force(self) -> None:
        (self.tmp_path / "reference.txt").write_text("old", encoding="utf-8")
        model_class, _, _ = make_model_class([FakeSegment(0.0, 1.0, "New text.")])

        exit_code, _, stderr = self._run([str(self.input)], model_class=model_class)
        self.assertEqual(exit_code, 2)
        self.assertIn("already exists", stderr)

        exit_code, _, _ = self._run(
            [str(self.input), "--force"], model_class=model_class
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            (self.tmp_path / "reference.txt").read_text(encoding="utf-8"),
            "New text.\n",
        )

    def test_missing_input_file_fails(self) -> None:
        exit_code, _, stderr = self._run(
            [str(self.tmp_path / "missing.wav")], model_class=make_model_class([])[0]
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("does not exist", stderr)

    def test_show_segments_prints_timestamps(self) -> None:
        model_class, _, _ = make_model_class([FakeSegment(0.0, 2.0, "Hello there.")])
        exit_code, stdout, _ = self._run(
            [str(self.input), "--show-segments"], model_class=model_class
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("[00:00 - 00:02]", stdout)

    def test_custom_output_path_creates_parent_directory(self) -> None:
        target = self.tmp_path / "nested" / "transcript.txt"
        model_class, _, _ = make_model_class([FakeSegment(0.0, 1.0, "Hello.")])
        exit_code, _, _ = self._run(
            [str(self.input), "-o", str(target)], model_class=model_class
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(target.exists())

    def test_device_and_compute_type_reach_model_loader(self) -> None:
        model_class, loads, _ = make_model_class([FakeSegment(0.0, 1.0, "Hello.")])
        exit_code, _, _ = self._run(
            [str(self.input), "--device", "cpu", "--compute-type", "float32"],
            model_class=model_class,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(loads, [("large-v3", "cpu", "float32")])

    def test_device_fallback_note_goes_to_stderr(self) -> None:
        model_class, _, _ = make_model_class(
            [FakeSegment(0.0, 1.0, "Hello.")], fail_load_devices=("cuda",)
        )
        exit_code, stdout, stderr = self._run(
            [str(self.input)], model_class=model_class
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Note: cuda", stderr)
        self.assertNotIn("Note: cuda", stdout)


if __name__ == "__main__":
    unittest.main()
