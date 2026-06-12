#!/usr/bin/env python3
"""Generate synthetic extreme OOD images for the limitation probe suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (SCRIPT_DIR.parent / "test_images").resolve()
DEFAULT_SOURCE_TEST = (SCRIPT_DIR.parent / "test_images" / "test.jpg").resolve()

SIZE = (512, 512)
NOISE_SEED = 42
NOISE_STD = 50
NOISY_EPSILON = 20
STEELBLUE = (70, 130, 180)


def pure_noise(path: Path) -> None:
    rng = np.random.default_rng(NOISE_SEED)
    arr = rng.integers(0, 256, size=(*SIZE, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, quality=95)


def solid_color(path: Path) -> None:
    Image.new("RGB", SIZE, color=STEELBLUE).save(path, quality=95)


def geometric_circle(path: Path) -> None:
    img = Image.new("RGB", SIZE, color="white")
    draw = ImageDraw.Draw(img)
    w, h = SIZE
    radius = min(w, h) // 4
    cx, cy = w // 2, h // 2
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill="black",
        outline="black",
    )
    img.save(path, quality=95)


def test_noisy(path: Path, source_test: Path) -> None:
    if not source_test.is_file():
        raise FileNotFoundError(f"Source test image not found: {source_test}")
    base = np.array(Image.open(source_test).convert("RGB"), dtype=np.int16)
    rng = np.random.default_rng(NOISE_SEED + 1)
    noise = rng.integers(-NOISY_EPSILON, NOISY_EPSILON + 1, size=base.shape, dtype=np.int16)
    noisy = np.clip(base + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy, mode="RGB").save(path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated images (default: ../test_images/)",
    )
    parser.add_argument(
        "--source-test",
        type=Path,
        default=DEFAULT_SOURCE_TEST,
        help="Clean test.jpg used to build test_noisy.jpg",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "pure_noise.jpg": lambda p: pure_noise(p),
        "solid_color.jpg": lambda p: solid_color(p),
        "geometric_circle.jpg": lambda p: geometric_circle(p),
        "test_noisy.jpg": lambda p: test_noisy(p, args.source_test),
    }

    for name, fn in outputs.items():
        out_path = args.output_dir / name
        fn(out_path)
        print(f"[OK] {out_path}")

    print(f"\nGenerated {len(outputs)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
