from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_audio.py"

_spec = importlib.util.spec_from_file_location("extract_audio", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
extract_audio = importlib.util.module_from_spec(_spec)
sys.modules["extract_audio"] = extract_audio
_spec.loader.exec_module(extract_audio)

ExtractError = extract_audio.ExtractError
AudioReport = extract_audio.AudioReport

try:
    FFMPEG = extract_audio.resolve_ffmpeg()
except ExtractError:
    FFMPEG = None

SAMPLE_BANNER = """
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Metadata:
    major_brand     : isom
  Duration: 00:00:06.02, start: 0.000000, bitrate: 181 kb/s
  Stream #0:0[0x1](und): Video: mpeg4 (Simple Profile) (mp4v / 0x7634706D), yuv420p, 160x120 [SAR 1:1 DAR 4:3], 10 fps, 10 tbr, 10240 tbn (default)
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, mono, fltp, 128 kb/s (default)
"""

TWO_AUDIO_BANNER = (
    "  Stream #0:0: Video: h264, yuv420p, 1280x720\n"
    "  Stream #0:1(eng): Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s\n"
    "  Stream #0:2(jpn): Audio: aac (LC), 48000 Hz, stereo, fltp, 192 kb/s\n"
)


class TimeParsingTest(unittest.TestCase):
    def test_accepts_seconds(self) -> None:
        self.assertAlmostEqual(extract_audio.parse_time("12.5"), 12.5)
        self.assertAlmostEqual(extract_audio.parse_time(" 0 "), 0.0)

    def test_accepts_minutes_and_hours(self) -> None:
        self.assertAlmostEqual(extract_audio.parse_time("1:30"), 90.0)
        self.assertAlmostEqual(extract_audio.parse_time("0:01:02.5"), 62.5)

    def test_rejects_invalid_values(self) -> None:
        for value in ("", "abc", "-4", "1:2:3:4"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    extract_audio.parse_time(value)


class CommandBuildingTest(unittest.TestCase):
    def test_default_extract_command_covers_cloning_needs(self) -> None:
        command = extract_audio.build_extract_command(
            "ffmpeg", Path("in.mp4"), Path("out.wav")
        )
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("-nostdin", command)
        self.assertIn("0:a:0", command)
        self.assertIn(extract_audio.LOUDNESS_FILTER, command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        self.assertNotIn("-ar", command)

    def test_seek_precedes_input_and_duration_follows(self) -> None:
        command = extract_audio.build_extract_command(
            "ffmpeg",
            Path("in.mp4"),
            Path("out.wav"),
            start_seconds=12.0,
            duration_seconds=8.5,
        )
        self.assertLess(command.index("-ss"), command.index("-i"))
        self.assertGreater(command.index("-t"), command.index("-i"))
        self.assertEqual(command[command.index("-ss") + 1], "12")
        self.assertEqual(command[command.index("-t") + 1], "8.5")

    def test_sample_rate_and_no_normalize_flags(self) -> None:
        command = extract_audio.build_extract_command(
            "ffmpeg",
            Path("in.mp4"),
            Path("out.wav"),
            channels=2,
            sample_rate=16000,
            normalize=False,
        )
        self.assertNotIn("-af", command)
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-ac") + 1], "2")

    def test_copy_command_uses_stream_copy(self) -> None:
        command = extract_audio.build_copy_command(
            "ffmpeg", Path("in.mp4"), Path("audio.m4a")
        )
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertNotIn("-af", command)

    def test_output_extension_rules(self) -> None:
        with self.assertRaises(ExtractError):
            extract_audio._validate_output_extension(Path("out.m4a"), copy=False)
        with self.assertRaises(ExtractError):
            extract_audio._validate_output_extension(Path("out.wav"), copy=True)
        extract_audio._validate_output_extension(Path("out.WAV"), copy=False)
        extract_audio._validate_output_extension(Path("out.mka"), copy=True)


class BannerParsingTest(unittest.TestCase):
    def test_reads_sample_rates_from_banner(self) -> None:
        self.assertEqual(extract_audio._sample_rates_from_banner(SAMPLE_BANNER), [48000])

    def test_reads_multiple_audio_streams(self) -> None:
        self.assertEqual(
            extract_audio._sample_rates_from_banner(TWO_AUDIO_BANNER), [44100, 48000]
        )

    def test_ignores_banner_without_audio(self) -> None:
        self.assertEqual(extract_audio._sample_rates_from_banner("Stream #0:0: Video"), [])


class ProbeParsingTest(unittest.TestCase):
    def test_probe_streams_parses_ffprobe_json(self) -> None:
        payload = (
            '{"streams": [{"codec_type": "audio", "codec_name": "aac", '
            '"sample_rate": "48000", "channels": 1, "bit_rate": "128000"}]}'
        )
        completed = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=0, stdout=payload, stderr=""
        )
        with mock.patch.object(extract_audio.subprocess, "run", return_value=completed):
            streams = extract_audio.probe_streams("ffprobe", Path("clip.mp4"))

        self.assertIsNotNone(streams)
        assert streams is not None
        self.assertEqual(streams[0]["codec_name"], "aac")
        self.assertEqual(
            extract_audio.describe_audio_stream(streams[0]),
            "aac, 48000 Hz, 1 ch, 128 kb/s",
        )

    def test_probe_streams_returns_none_on_failure(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ffprobe"], returncode=1, stdout="", stderr="boom"
        )
        with mock.patch.object(extract_audio.subprocess, "run", return_value=completed):
            self.assertIsNone(extract_audio.probe_streams("ffprobe", Path("clip.mp4")))

    def test_print_streams_labels_audio_and_video(self) -> None:
        streams = [
            {"codec_type": "video", "codec_name": "h264", "width": 640, "height": 360},
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
            },
        ]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            extract_audio.print_streams(streams)

        text = stdout.getvalue()
        self.assertIn("v:0", text)
        self.assertIn("a:0", text)
        self.assertIn("a:1", text)
        self.assertIn("640x360", text)


class BinaryResolutionTest(unittest.TestCase):
    def test_explicit_ffmpeg_path_is_used(self) -> None:
        self.assertEqual(
            extract_audio.resolve_ffmpeg(sys.executable), sys.executable
        )

    def test_missing_explicit_ffmpeg_path_fails(self) -> None:
        with self.assertRaises(ExtractError):
            extract_audio.resolve_ffmpeg(str(Path("missing-ffmpeg.exe")))


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def _write_wav(self, samples: np.ndarray, sample_rate: int = 48000) -> Path:
        path = self.tmp_path / "sample.wav"
        sf.write(str(path), samples, sample_rate, subtype="PCM_16")
        return path

    def test_reports_sine_metrics(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2) / sample_rate
        report = extract_audio.analyze_audio_file(
            self._write_wav(0.5 * np.sin(2 * np.pi * 440 * time), sample_rate)
        )
        self.assertIsNotNone(report)
        assert report is not None
        self.assertAlmostEqual(report.duration_seconds, 2.0, places=2)
        self.assertEqual(report.sample_rate, sample_rate)
        self.assertEqual(report.channels, 1)
        self.assertAlmostEqual(report.peak_dbfs, -6.0, places=1)
        self.assertEqual(report.clipped_samples, 0)
        self.assertLess(report.silence_ratio, 0.01)

    def test_reports_silence(self) -> None:
        report = extract_audio.analyze_audio_file(self._write_wav(np.zeros(4800)))
        self.assertIsNotNone(report)
        assert report is not None
        self.assertIsNone(report.peak_dbfs)
        self.assertAlmostEqual(report.silence_ratio, 1.0)

    def test_warns_outside_reference_range(self) -> None:
        def report_for(duration: float) -> AudioReport:
            return AudioReport(
                duration_seconds=duration,
                sample_rate=48000,
                channels=1,
                subtype="PCM_16",
                peak_dbfs=-6.0,
                rms_dbfs=-12.0,
                clipped_samples=0,
                dc_offset=0.0,
                silence_ratio=0.0,
            )

        short = extract_audio.reference_warnings(report_for(0.5), normalize=True)
        self.assertTrue(any("only" in note for note in short))

        outside = extract_audio.reference_warnings(report_for(15.0), normalize=True)
        self.assertTrue(any("outside the recommended" in note for note in outside))

        long_clip = extract_audio.reference_warnings(report_for(45.0), normalize=True)
        self.assertTrue(any("3-10 s excerpt" in note for note in long_clip))

        clean = extract_audio.reference_warnings(report_for(6.0), normalize=True)
        self.assertEqual(clean, [])


@unittest.skipUnless(FFMPEG, "FFmpeg is not available")
class ExtractionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._temp.name)
        cls.addClassCleanup(cls._temp.cleanup)

        cls.clip = cls.tmp_path / "clip.mp4"
        extract_audio.run_command(
            [
                FFMPEG or "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6:sample_rate=48000",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=6:size=160x120:rate=10",
                "-map",
                "0:a",
                "-map",
                "1:v",
                "-c:a",
                "aac",
                "-c:v",
                "mpeg4",
                "-shortest",
                str(cls.clip),
            ]
        )

        cls.silent = cls.tmp_path / "silent.mp4"
        extract_audio.run_command(
            [
                FFMPEG or "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=160x120:rate=10",
                "-c:v",
                "mpeg4",
                str(cls.silent),
            ]
        )

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = extract_audio.main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_extracts_mono_wav_keeping_source_rate(self) -> None:
        output = self.tmp_path / "ref.wav"
        exit_code, stdout, _ = self._run(
            [str(self.clip), "--duration", "4", "-o", str(output)]
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Wrote reference clip", stdout)

        info = sf.info(str(output))
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, 1)
        self.assertAlmostEqual(info.duration, 4.0, delta=0.2)

    def test_honors_sample_rate_flag(self) -> None:
        output = self.tmp_path / "ref16.wav"
        exit_code, _, _ = self._run(
            [str(self.clip), "--duration", "3", "--sample-rate", "16000", "-o", str(output)]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(sf.info(str(output)).samplerate, 16000)

    def test_rejects_video_only_input(self) -> None:
        output = self.tmp_path / "nope.wav"
        exit_code, _, stderr = self._run([str(self.silent), "-o", str(output)])
        self.assertEqual(exit_code, 2)
        self.assertIn("no audio stream", stderr)
        self.assertFalse(output.exists())

    def test_refuses_to_overwrite_without_force(self) -> None:
        output = self.tmp_path / "guard.wav"
        first, _, _ = self._run([str(self.clip), "--duration", "2", "-o", str(output)])
        second, _, stderr = self._run(
            [str(self.clip), "--duration", "2", "-o", str(output)]
        )
        self.assertEqual(first, 0)
        self.assertEqual(second, 2)
        self.assertIn("already exists", stderr)

    def test_copy_mode_requires_output_and_copies_stream(self) -> None:
        exit_code, _, stderr = self._run([str(self.clip), "--copy"])
        self.assertEqual(exit_code, 2)
        self.assertIn("--output", stderr)

        copied = self.tmp_path / "audio.m4a"
        exit_code, _, _ = self._run([str(self.clip), "--copy", "-o", str(copied)])
        self.assertEqual(exit_code, 0)
        self.assertTrue(copied.exists())
        self.assertNotEqual(copied.read_bytes()[:4], b"RIFF")

    def test_validates_duration_and_stream_index(self) -> None:
        output = self.tmp_path / "bad.wav"
        exit_code, _, _ = self._run(
            [str(self.clip), "--duration", "0", "-o", str(output)]
        )
        self.assertEqual(exit_code, 2)

        exit_code, _, stderr = self._run(
            [str(self.clip), "--stream-index", "3", "-o", str(output)]
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("out of range", stderr)

    def test_missing_input_file_fails(self) -> None:
        exit_code, _, stderr = self._run(
            [str(self.tmp_path / "missing.mp4"), "-o", str(self.tmp_path / "x.wav")]
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("does not exist", stderr)

    def test_copy_conflicts_with_tuning_flags(self) -> None:
        exit_code, _, stderr = self._run(
            [str(self.clip), "--copy", "-o", str(self.tmp_path / "x.m4a"), "--no-normalize"]
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("cannot be combined", stderr)


if __name__ == "__main__":
    unittest.main()
