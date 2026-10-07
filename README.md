# ZeroShot-VoiceClone

Command-line voice cloning MVP with a pluggable backend architecture.

## What It Does

The scaffold in this repository provides a single CLI command that:

- takes a reference audio sample
- takes a transcript for that reference sample, unless reduced-quality x-vector mode is requested
- reads a target text file
- synthesizes an output audio file in the cloned voice
- writes a JSON sidecar with the generation settings

The shipped backends are currently `qwen` and `xtts`. Qwen defaults to `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, while XTTS defaults to `coqui/XTTS-v2` through Coqui's TTS API.

## Status

This is an MVP scaffold. It is designed to be a clean starting point, not a production service. You will still need a working Qwen3-TTS runtime environment and a GPU-capable machine for practical inference speeds.

## Install

Qwen recommends using a fresh Python 3.12 environment for runtime. The following setup commands are for Windows PowerShell, run from the project root. Check that `python --version` reports your intended Python version before creating the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore -e .
```

These setup commands target `.venv` explicitly and do not require activation. For the later `python` and `voiceclone` examples, activate it with `.\.venv\Scripts\Activate.ps1`. If activation is blocked by your PowerShell execution policy, use `.\.venv\Scripts\python.exe` instead of `python`, and `.\.venv\Scripts\python.exe -m zero_shot_voiceclone` instead of `voiceclone`.

### Setup checklist

| Component | When you need it | Installation |
| --- | --- | --- |
| Qwen and Python audio dependencies | Qwen synthesis | Main install above |
| CUDA-enabled PyTorch and torchaudio | NVIDIA GPU acceleration | Matching wheels below; CPU inference also works |
| Native SoX executable | Qwen audio-processing paths that call SoX | Separate Windows install below; pip installs only the wrapper |
| FFmpeg | Extracting reference clips from video | System FFmpeg or the `extract` extra |
| faster-whisper | Generating a reference transcript | The `transcribe` extra |
| `truststore` | Model downloads that fail with SSL certificate errors | Certificate troubleshooting below |
| `flash-attn` | Optional inference optimization | Not required to start synthesis |

Install the main package and any optional extras first, then install the CUDA wheels if you need GPU acceleration. Resolve SoX/PATH and certificate issues as described below before starting a long synthesis run.

If you want the XTTS-v2 backend as well, install the optional dependency set:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore -e ".[xtts]"
```

That extra now includes Coqui's `codec` support, which recent XTTS builds need for audio I/O.

### GPU acceleration (recommended)

The project may initially install CPU-only PyTorch. For NVIDIA GPU inference, install matching CUDA-enabled PyTorch and torchaudio wheels **after** the main install and any optional extras. Run this from the project root in PowerShell to target the project's `.venv` explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchaudio==2.11.0+cu128
```

This CUDA 12.8 setup was verified on an NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU. It requires a compatible NVIDIA driver. The wheels bundle cuBLAS and cuDNN, including `cublas64_12.dll` and `cudnn64_9.dll`, so a separate CUDA Toolkit or standalone DLL download is not needed. `torchvision` is not required for this setup.

The pip truststore option uses system certificate trust and keeps HTTPS verification enabled, including on networks where Python's certificate bundle causes download errors.

Verify CUDA is available:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output: `2.11.0+cu128 12.8 True`. If it reports `+cpu`, `None`, or `False`, check that you are using the project venv and that the NVIDIA driver supports CUDA 12.8.

Both backends auto-detect CUDA when `--device auto` (the default) is used. You can also pass `--device cuda` explicitly.

FlashAttention is optional. The startup message `Warning: flash-attn is not installed. Will only run the manual PyTorch version.` does not prevent synthesis or mean CUDA is unavailable. Start with the PyTorch fallback; only install FlashAttention separately if you have a compatible build for your Windows, Python, PyTorch, CUDA, and GPU combination.

### SoX (for Qwen audio processing)

The Python `sox` package is a wrapper, not the native `sox.exe` program. Qwen imports the wrapper at startup, which checks whether the executable is on PATH and prints `SoX could not be found!` when it is missing. Installing the Python package again will not fix a missing executable.

The default 12Hz Qwen models use `librosa` for reference resampling, so this import-time warning alone does not mean synthesis failed. Other Qwen audio-processing paths invoke SoX and need the native program. For full runtime support on Windows, install it via:

```powershell
winget install sox
```

Ensure the directory containing `sox.exe` is on your user PATH, then open a new shell. Restart VS Code as well if its integrated terminals still inherit the old PATH. Verify the native command is available:

```powershell
sox --version
```

If it still reports `'sox' is not recognized`, add the installation directory to PATH, not the path to `sox.exe` itself, and restart the shell again.

### FFmpeg (for extracting reference clips from video)

[scripts/extract_audio.py](scripts/extract_audio.py) shells out to FFmpeg when you want to pull a reference clip out of an MP4 or another media file. Install FFmpeg and make sure `ffmpeg` is on PATH:

```powershell
winget install Gyan.FFmpeg
```

Alternatively, install the optional bundled-binary fallback, which the script detects automatically:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore -e ".[extract]"
```

### faster-whisper (for transcribing reference clips)

[scripts/transcribe_audio.py](scripts/transcribe_audio.py) turns a reference clip into the exact transcript file the Qwen backend needs. It uses faster-whisper, which is an optional dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore -e ".[transcribe]"
```

On first use the script downloads the selected model (~3 GB for the default `large-v3`) into the Hugging Face cache. When PyTorch is installed with CUDA support, the script reuses its bundled CUDA libraries, so GPU transcription works without installing a separate CUDA toolkit.

### Verify the Python setup

Run these checks without downloading a model or starting synthesis:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m zero_shot_voiceclone --help
```

Expect `No broken requirements found.` and the CLI help output. These checks verify package consistency and the CLI entry point, not model loading or audio generation. Use the CUDA and SoX checks above to verify those separately. The first synthesis run may still need to download model weights; certificate troubleshooting below applies to both Qwen and faster-whisper downloads.

## Extracting a reference clip from a video

`scripts/extract_audio.py` decodes an MP4 (or any media file FFmpeg can read) to a mono PCM WAV in a single decode pass, so the audio is never pushed through a second lossy stage:

```bash
python scripts/extract_audio.py interview.mp4 --start 12 --duration 8
```

The script keeps the source sample rate by default, downmixes to mono, applies single-pass loudness normalization to -16 LUFS, and prints a quality report with warnings when the clip falls outside the recommended 3 to 10 second reference range. Useful flags:

- `--sample-rate 16000` forces a specific rate; nothing is upsampled unless you ask.
- `--no-normalize` keeps the source loudness.
- `--list-streams` and `--stream-index 1` inspect and pick one of several audio tracks in a multi-language MP4.
- `--copy -o audio.m4a` copies the compressed stream without re-encoding, for archival.
- `--force` overwrites an existing output file.

Use the extracted WAV as `--reference-audio`. Trim the exact sentence you want to clone and save its transcript to a text file for `--reference-text-file`.

## Transcribing a reference clip

`scripts/transcribe_audio.py` produces the `--reference-text-file` content for the best-quality Qwen clone mode. It accepts audio or video containers, so you can transcribe the MP4 directly or, better, the exact clip you already extracted:

```bash
python scripts/transcribe_audio.py reference.trimmed.wav --language en
```

The transcript is written to `<input stem>.txt` and printed. Useful flags:

- `--model large-v3` selects the faster-whisper model (`tiny.en`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`, or a local path). The default is `large-v3`.
- `--language en` sets the spoken language explicitly; the default auto-detects.
- `--show-segments` prints sentence timestamps, handy for choosing the `--start` and `--duration` values for `extract_audio.py`.
- `--device cuda` and `--compute-type float16` control where and how the model runs. `auto`, the default, tries CUDA first and falls back to CPU with a note.
- `--no-vad` disables voice-activity detection, which is on by default to avoid hallucinated text on silence.

### Windows SSL certificate errors during model downloads

If a Hugging Face model download fails with `SSLCertVerificationError`, install `truststore` in the same Python environment used to run the project. The pip truststore option also helps if pip itself has a certificate-bundle problem:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore truststore
```

Inject the Windows certificate store before starting the transcriber (replace the example audio path as needed):

```powershell
.\.venv\Scripts\python.exe -c "import runpy, sys, truststore; truststore.inject_into_ssl(); sys.argv = ['scripts/transcribe_audio.py', r'reference.trimmed.wav']; runpy.run_path(sys.argv[0], run_name='__main__')"
```

For a Qwen model download during synthesis, inject `truststore` before importing the CLI:

```powershell
.\.venv\Scripts\python.exe -c "import sys, truststore; truststore.inject_into_ssl(); from zero_shot_voiceclone.cli import main; sys.argv = ['voiceclone', 'synth', '--backend', 'qwen', '--reference-audio', 'sample.wav', '--reference-text-file', 'sample.txt', '--target-text-file', 'script.txt', '--output', 'out.wav', '--confirm-rights-to-voice']; raise SystemExit(main())"
```

These commands keep HTTPS certificate verification enabled and use certificates trusted by Windows. If the error persists, the certificate authority may not be trusted by Windows; try a network that provides a valid public certificate chain. Do not disable SSL verification. Setting `HF_HUB_DISABLE_XET=1` alone does not fix certificate validation.

If transcription fails with `open() got an unexpected keyword argument 'metadata_errors'`, an existing environment may have PyAV 19, which is incompatible with the argument used by faster-whisper. The `transcribe` extra constrains PyAV to a compatible version; refresh an existing environment with:

```powershell
.\.venv\Scripts\python.exe -m pip install --use-feature=truststore -e ".[transcribe]"
```

For the best clone quality, extract the clip first and then transcribe that exact clip, so transcript and audio match word for word. Verify unusual words, names, and numbers by ear before cloning.

## Usage

Examples below are shown as single-line commands so they work directly in PowerShell and Bash.

Qwen clone mode with a transcript for the reference clip:

```bash
voiceclone synth --backend qwen --reference-audio sample.wav --reference-text-file sample.txt --target-text-file script.txt --output out.wav --confirm-rights-to-voice
```

Qwen reduced-quality x-vector-only mode without a reference transcript:

```bash
voiceclone synth --backend qwen --reference-audio sample.wav --target-text-file script.txt --output out.wav --qwen-x-vector-only --confirm-rights-to-voice
```

XTTS-v2 clone mode with an explicit language code:

```bash
voiceclone synth --backend xtts --reference-audio sample.wav --target-text-file script.txt --output out.wav --language en --confirm-rights-to-voice
```

For the best Qwen cloning results, use a short trimmed mono WAV plus a transcript file that matches the spoken reference clip exactly. For example, if your trimmed clip says this sentence:

```text
On a bright autumn morning, I walked past the quiet market and watched the pale blue sky above the town.
```

save that exact line into `reference-qwen.txt`, then run:

```bash
voiceclone synth --backend qwen --model "Qwen/Qwen3-TTS-12Hz-1.7B-Base" --reference-audio reference.trimmed.wav --reference-text-file reference-qwen.txt --target-text-file script.txt --output out-qwen.wav --language Auto --confirm-rights-to-voice
```

Shared flags cover the reference audio, target text file, output path, model, language, chunking, metadata, and consent acknowledgement. Backend-specific flags are now prefixed, for example `--qwen-device` or `--xtts-split-sentences`.

The repository includes [script.txt](script.txt) as a sample target text file for smoke tests. The `sample.wav` and `sample.txt` names in the examples are placeholders and should be replaced with your own reference audio and transcript.

The CLI defaults to `--backend qwen`.

### Qwen performance controls

Qwen generation is limited to 2,048 new codec tokens per chunk by default, rather than inheriting the model's potentially much larger limit (8,192 in the default 0.6B model). This bounds excessively long generation for short text; it does not make normally terminating generation intrinsically faster.

Use `--qwen-max-new-tokens` to override the positive per-chunk limit. A lower limit can cut off speech, especially with long chunks or slow delivery, so increase it if output is truncated. Use `--qwen-max-new-tokens 8192` to restore the downloaded model's previous ceiling when needed. The chosen limit is recorded in the output metadata.

Larger text chunks reduce repeated generation calls, though each call may take longer and use more memory. For the bundled script, `--chunk-max-chars 360` produces 14 chunks instead of 22 and avoids the isolated one-word chunk produced at 240 characters. The shared default remains 240 characters.

An English synthesis command using the shortened local reference and a 1,024-token ceiling is:

```powershell
.\.venv\Scripts\python.exe -c "import truststore; truststore.inject_into_ssl(); from zero_shot_voiceclone.cli import main; raise SystemExit(main())" synth --backend qwen --qwen-device cuda:0 --language English --reference-audio ".\sample_voice\Mohideen.short.wav" --reference-text-file ".\sample_voice\Mohideen.short.txt" --target-text-file ".\script.txt" --chunk-max-chars 360 --qwen-max-new-tokens 1024 --output ".\out-mohideen.wav" --confirm-rights-to-voice
```

Code changes and new flags require a fresh CLI invocation; a running Python process does not automatically reload them.

## Notes

- Reference audio is best kept short and clean. The CLI warns when it falls outside the recommended 3 to 10 second range, including when it exceeds 30 seconds, but does not truncate it. Trim a clean speech excerpt yourself and keep the transcript matched to that exact excerpt.
- Qwen quality improves substantially when the reference clip is a single clean 5 to 8 second speech segment in mono WAV format and the transcript matches the clip word for word.
- Long target documents are split into sentence-sized chunks and stitched back together with a configurable silence gap.
- XTTS-v2 expects an explicit language code such as `en`, `es`, or `zh-cn`; it does not support `Auto`.
- On first XTTS-v2 model download, Coqui prompts for CPML/commercial-license acknowledgement. If you have already reviewed and accepted those terms, you can avoid the interactive prompt by setting `COQUI_TOS_AGREED=1` in the shell that launches the CLI.
- The tool refuses to run unless you pass `--confirm-rights-to-voice`.

## Project Layout

```text
docs/mvp-plan.md                Implementation plan
scripts/extract_audio.py        Reference-clip extractor for video files
scripts/transcribe_audio.py     Reference-clip transcriber (faster-whisper)
src/zero_shot_voiceclone/backends/
src/zero_shot_voiceclone/       CLI package
tests/                          Lightweight unit tests
```