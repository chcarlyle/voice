import argparse
from itertools import count
from pathlib import Path
import sys
import textwrap
import time

import soundfile as sf
import torch

from chatterbox.tts import ChatterboxTTS


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def patch_torch_load(device: torch.device) -> None:
    original_load = torch.load

    def patched(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = device
        return original_load(*args, **kwargs)

    torch.load = patched  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a voice prompt once, then repeatedly synthesize new text with Chatterbox TTS."
    )
    parser.add_argument(
        "--voice-prompt",
        required=True,
        help="Path to the reference clip whose voice should be cloned.",
    )
    parser.add_argument(
        "--text",
        help="Optional single text string to synthesize. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_audio",
        help="Directory where synthesized wav files are written (default: %(default)s).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="utterance",
        help="Prefix for output filenames (default: %(default)s).",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion scaling factor passed to `prepare_conditionals` and `generate` (default: %(default)s).",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.25,
        help="Classifier-free guidance weight for T3 sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for T3 inference (default: %(default)s).",
    )
    return parser.parse_args()


def generate_and_save(
    tts: ChatterboxTTS,
    text: str,
    out_path: Path,
    *,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
) -> None:
    start = time.perf_counter()
    wav = tts.generate(
        text,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start
    sf.write(out_path, wav.squeeze(0).cpu().numpy(), tts.sr)
    print(f"[saved] {out_path} ({elapsed:.2f}s)")


def interactive_loop(
    tts: ChatterboxTTS,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    intro = textwrap.dedent(
        """
        Enter text to synthesize (blank line or Ctrl-D to exit).
        The cached voice prompt will be reused for every line.
        """
    ).strip()
    print(intro)
    counter = count(1)
    try:
        while True:
            prompt = input("Text> ").strip()
            if not prompt:
                print("Empty input detected. Exiting.")
                break
            idx = next(counter)
            out_path = output_dir / f"{args.filename_prefix}_{idx:03d}.wav"
            generate_and_save(
                tts,
                prompt,
                out_path,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
            )
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")


def main() -> None:
    args = parse_args()
    voice_prompt = Path(args.voice_prompt).expanduser().resolve()
    if not voice_prompt.exists():
        print(f"Reference clip not found: {voice_prompt}", file=sys.stderr)
        sys.exit(1)

    device = detect_device()
    print(f"Using device: {device}")
    patch_torch_load(torch.device("cpu") if device.type in {"cpu", "mps"} else device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ChatterboxTTS weights...")
    tts = ChatterboxTTS.from_pretrained(device=device.type)
    print(f"Preparing conditionals from {voice_prompt.name}...")
    tts.prepare_conditionals(str(voice_prompt), exaggeration=args.exaggeration)
    print("Voice cached. Ready to synthesize!")

    if args.text:
        out_path = output_dir / f"{args.filename_prefix}_001.wav"
        generate_and_save(
            tts,
            args.text,
            out_path,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
        )
    else:
        interactive_loop(tts, args, output_dir)


if __name__ == "__main__":
    main()
