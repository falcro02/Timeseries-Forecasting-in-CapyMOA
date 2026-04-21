from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from capymoa.datasets import Bike
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import AdaptiveRandomForestRegressor, FIMTDD, SGDRegressor
from capymoa.stream import NumpyStream
import numpy as np
import matplotlib.pyplot as plt

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ForecastDatasetBuilder, LagTransformer


def metric_as_float(value: Any) -> float:
    """Convert evaluator metric output to a single float value.

    CapyMOA cumulative evaluators usually return a scalar, while windowed
    evaluators can return a list/array of values across windows.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float("nan")
        return float(value[-1])
    return float(value)


def metric_as_series(value: Any) -> list[float]:
    """Convert evaluator metric output to a list for plotting."""
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def window_end_samples(num_points: int, window_size: int, total_samples: int) -> np.ndarray:
    """Map each windowed metric point to the corresponding sample count."""
    return np.array(
        [min((idx + 1) * window_size, total_samples) for idx in range(num_points)],
        dtype=int,
    )


def build_model(name: str, schema, learning_rate: float, random_seed: int):
    if name == "sgd":
        return SGDRegressor(
            schema=schema,
            learning_rate="invscaling",
            eta0=learning_rate,
            penalty="l2",
            alpha=0.0001,
            random_seed=random_seed,
        )
    if name == "arf":
        return AdaptiveRandomForestRegressor(schema=schema, random_seed=random_seed)
    if name == "fimtdd":
        return FIMTDD(schema=schema, random_seed=random_seed)
    raise ValueError(f"Unknown model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 3: prequential simulation on transformed Bike forecasting samples.",
    )
    parser.add_argument("--lag-size", type=int, default=24, help="Lag window size k.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecasting horizon H.")
    parser.add_argument(
        "--include-input-lags",
        action="store_true",
        help="Include lagged input features in addition to lagged target.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum transformed samples to evaluate. Use -1 for all.",
    )
    parser.add_argument(
        "--model",
        choices=["sgd", "arf", "fimtdd"],
        default="sgd",
        help="CapyMOA regressor used in prequential evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate used by sgd model.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size used by CapyMOA prequential evaluator.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Random seed used by learners that support it.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create MAE/RMSE windowed plots (default: true).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the plot window interactively.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/prequential_bike_metrics.png",
        help="Path of the saved metrics plot image.",
    )
    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Disable saving the plot image to disk.",
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
        ),
    )

    samples = builder.build_forecasting_dataset(
        source=stream,
        horizon=args.horizon,
        max_samples=max_samples,
    )

    if not samples:
        print("No forecasting samples available for evaluation.")
        return

    x_data = np.array([sample.features for sample in samples], dtype=float)
    y_data = np.array([sample.target for sample in samples], dtype=float)

    transformed_stream = NumpyStream(
        x_data,
        y_data,
        dataset_name="BikeForecastingTransformed",
        target_type="numeric",
    )

    learner = build_model(
        name=args.model,
        schema=transformed_stream.get_schema(),
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    results = prequential_evaluation(
        stream=transformed_stream,
        learner=learner,
        max_instances=len(samples),
        window_size=args.window_size,
        optimise=True,
        progress_bar=False,
    )

    cumulative = results["cumulative"]
    windowed = results["windowed"]

    mode = "one-step" if args.horizon == 1 else f"aggregated H={args.horizon}"
    print("\n=== Prequential Evaluation (Bike) ===")
    print(f"mode: {mode}")
    print(f"model: {args.model}")
    print(f"lag size (k): {args.lag_size}")
    print(f"include input lags: {args.include_input_lags}")
    print(f"evaluated samples: {len(samples)}")
    if hasattr(cumulative, "mae"):
        print(f"cumulative MAE: {metric_as_float(cumulative.mae()):.4f}")
    print(f"cumulative RMSE: {metric_as_float(cumulative.rmse()):.4f}")
    if hasattr(windowed, "mae"):
        print(f"windowed MAE: {metric_as_float(windowed.mae()):.4f}")
    print(f"windowed RMSE: {metric_as_float(windowed.rmse()):.4f}")

    windowed_mae = metric_as_series(windowed.mae()) if hasattr(windowed, "mae") else []
    windowed_rmse = metric_as_series(windowed.rmse())

    if args.plot:
        x_rmse = window_end_samples(len(windowed_rmse), args.window_size, len(samples))
        plt.figure(figsize=(10, 5))

        if windowed_mae:
            x_mae = window_end_samples(len(windowed_mae), args.window_size, len(samples))
            plt.plot(x_mae, windowed_mae, label="Windowed MAE", linewidth=2)
        plt.plot(x_rmse, windowed_rmse, label="Windowed RMSE", linewidth=2)

        plt.title(f"Prequential Metrics on Bike ({mode}, model={args.model})")
        plt.xlabel("Number of evaluated samples")
        plt.ylabel("Metric value (MAE / RMSE)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if not args.no_save_plot:
            output_path = Path(args.save_plot)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=140)
            print(f"saved plot: {output_path}")

        if args.show_plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()
