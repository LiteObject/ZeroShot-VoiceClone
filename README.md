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

Qwen recommends using a fresh Python 3.12 environment for runtime. The project itself is packaged as a standard Python CLI:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

If you want the XTTS-v2 backend as well, install the optional dependency set:

```bash
pip install -e .[xtts]
```

That extra now includes Coqui's `codec` support, which recent XTTS builds need for audio I/O.

If you want the recommended FlashAttention path on compatible NVIDIA hardware, install it separately in that environment for Qwen.

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

Shared flags cover the reference audio, target text file, output path, model, language, chunking, metadata, and consent acknowledgement. Backend-specific flags are now prefixed, for example `--qwen-device` or `--xtts-split-sentences`.

The repository includes [script.txt](script.txt) as a sample target text file for smoke tests. The `sample.wav` and `sample.txt` names in the examples are placeholders and should be replaced with your own reference audio and transcript.

The CLI defaults to `--backend qwen`.

## Notes

- Reference audio is best kept short and clean. The CLI warns when it falls outside the recommended 3 to 10 second range and allows longer clips, though a shorter excerpt is usually better.
- Long target documents are split into sentence-sized chunks and stitched back together with a configurable silence gap.
- XTTS-v2 expects an explicit language code such as `en`, `es`, or `zh-cn`; it does not support `Auto`.
- On first XTTS-v2 model download, Coqui prompts for CPML/commercial-license acknowledgement. If you have already reviewed and accepted those terms, you can avoid the interactive prompt by setting `COQUI_TOS_AGREED=1` in the shell that launches the CLI.
- The tool refuses to run unless you pass `--confirm-rights-to-voice`.

## Project Layout

```text
docs/mvp-plan.md                Implementation plan
src/zero_shot_voiceclone/backends/
src/zero_shot_voiceclone/       CLI package
tests/                          Lightweight unit tests
```