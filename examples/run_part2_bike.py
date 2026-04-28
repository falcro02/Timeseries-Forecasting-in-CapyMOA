from pathlib import Path
import sys
import argparse

from capymoa.datasets import Bike

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ForecastingStream, LagTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one-step and aggregated-horizon forecasting samples on Bike."
    )
    parser.add_argument("--lag-size", type=int, default=24, help="Lag window size k.")
    parser.add_argument("--horizon", type=int, default=1, help="Aggregation horizon H.")
    parser.add_argument(
        "--include-input-lags",
        action="store_true",
        help="Include lagged input features in addition to lagged target.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of generated samples per mode. Use -1 for all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lag_size = args.lag_size
    horizon = args.horizon
    include_input_lags = args.include_input_lags
    max_samples = None if args.max_samples < 0 else args.max_samples

    stream = Bike()
    transformed_stream = ForecastingStream(
        source_stream=stream,
        transformer=LagTransformer(k=lag_size, include_input_lags=include_input_lags),
        horizon=horizon,
        max_samples=max_samples,
    )

    mode = "one-step" if horizon == 1 else f"aggregated H={horizon}"
    count = 0
    first_x = None
    first_y = None
    samples = []
    while transformed_stream.has_more_instances():
        inst = transformed_stream.next_instance()
        if count == 0:
            first_x = inst.x
            first_y = inst.y_value
        if len(samples) < 10:
            samples.append((inst.x, inst.y_value))
        count += 1

    print(f"Samples ({mode}): {count}")
    if count > 0:
        x_len = len(first_x) if (first_x is not None and hasattr(first_x, "__len__")) else 0
        target_val = float(first_y) if first_y is not None else float("nan")
        print(f"First sample -> x_len={x_len}, target={target_val:.3f}")
        print("\nFirst 10 samples (or fewer if stream shorter):")
        def _format_x(x):
            if x is None:
                return "None"
            try:
                lst = x.tolist() if hasattr(x, "tolist") else list(x) if hasattr(x, "__iter__") else x
            except Exception:
                return repr(x)
            if isinstance(lst, (list, tuple)):
                n = len(lst)
                if n > 20:
                    return f"{lst[:20]}...(+{n-20} more)"
                return repr(lst)
            return repr(lst)

        for i, (xx, yy) in enumerate(samples, start=1):
            try:
                yv = float(yy) if yy is not None else float("nan")
            except Exception:
                yv = repr(yy)
            print(f"#{i}: x={_format_x(xx)}, target={yv}")


if __name__ == "__main__":
    main()
