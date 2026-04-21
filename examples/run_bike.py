from pathlib import Path
import argparse
import sys

from capymoa.datasets import Bike

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ForecastDatasetBuilder, LagTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and print a transformed Bike forecasting dataset.",
    )
    parser.add_argument("--lag-size", type=int, default=2, help="Lag window size k.")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon H.")
    parser.add_argument(
        "--include-input-lags",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include lagged Bike input features (default: false).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum transformed samples to build. Use -1 for all.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="How many transformed samples to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = None if args.max_samples < 0 else args.max_samples

    stream = Bike()
    builder = ForecastDatasetBuilder(
        transformer=LagTransformer(
            k=args.lag_size,
            include_input_lags=args.include_input_lags,
        )
    )
    dataset = builder.build_forecasting_dataset(
        source=stream,
        horizon=args.horizon,
        max_samples=max_samples,
    )

    mode = "one-step" if args.horizon == 1 else f"aggregated H={args.horizon}"
    print("\n=== Transformed Bike Dataset ===")
    print(f"mode: {mode}")
    print(f"lag size (k): {args.lag_size}")
    print(f"include input lags: {args.include_input_lags}")
    print(f"samples built: {len(dataset)}")

    if not dataset:
        print("No samples generated. Increase data or reduce lag/horizon.")
        return

    show = max(0, min(args.show_samples, len(dataset)))
    print(f"\nFirst {show} samples:")
    for idx, sample in enumerate(dataset[:show], start=1):
        print(f"[{idx}] x_len={len(sample.features)}")
        print(f"    x={sample.features}")
        print(f"    y={sample.target:.3f}")


if __name__ == "__main__":
    main()
