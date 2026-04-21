from pathlib import Path
import sys
import argparse

from capymoa.datasets import Fried

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ForecastDatasetBuilder, LagTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one-step and aggregated-horizon forecasting samples on Fried."
    )
    parser.add_argument("--lag-size", type=int, default=1, help="Lag window size k.")
    parser.add_argument("--horizon", type=int, default=1, help="Aggregation horizon H.")
    parser.add_argument(
        "--include-input-lags",
        action="store_true",
        help="Include lagged input features in addition to lagged target.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of generated samples per mode. Use -1 for all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lag_size = args.lag_size
    horizon = args.horizon
    include_input_lags = args.include_input_lags
    max_samples = None if args.max_samples < 0 else args.max_samples

    stream = Fried()
    builder = ForecastDatasetBuilder(
        transformer=LagTransformer(k=lag_size, include_input_lags=include_input_lags),
    )
    samples = builder.build_forecasting_dataset(
        source=stream,
        horizon=horizon,
        max_samples=max_samples,
    )

    mode = "one-step" if horizon == 1 else f"aggregated H={horizon}"
    print(f"Samples ({mode}): {len(samples)}")
    if samples:
        first = samples[0]
        print(
            f"First sample -> x_len={len(first.features)}, target={first.target:.3f}"
        )


if __name__ == "__main__":
    main()
