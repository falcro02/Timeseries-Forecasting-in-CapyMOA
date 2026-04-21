from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from capymoa.datasets import Bike
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import AdaptiveRandomForestRegressor, FIMTDD, SGDRegressor


def metric_as_float(value: Any) -> float:
    """Convert evaluator metric output to a single float value."""
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
        description="Prequential evaluation on the raw Bike stream (no forecasting transform).",
    )
    parser.add_argument(
        "--model",
        choices=["sgd", "arf", "fimtdd"],
        default="arf",
        help="CapyMOA regressor used in prequential evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of Bike instances to evaluate. Use -1 for all.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size used by CapyMOA prequential evaluator.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate used by sgd model.",
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
        default="outputs/raw_bike_metrics.png",
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
    schema = stream.get_schema()
    learner = build_model(
        name=args.model,
        schema=schema,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    results = prequential_evaluation(
        stream=stream,
        learner=learner,
        max_instances=max_samples,
        window_size=args.window_size,
        optimise=True,
        progress_bar=False,
    )

    cumulative = results["cumulative"]
    windowed = results["windowed"]

    print("\n=== Prequential Evaluation (Raw Bike) ===")
    print(f"model: {args.model}")
    print(f"max samples: {max_samples if max_samples is not None else 'all'}")
    print(f"window size: {args.window_size}")
    if hasattr(cumulative, "mae"):
        print(f"cumulative MAE: {metric_as_float(cumulative.mae()):.4f}")
    print(f"cumulative RMSE: {metric_as_float(cumulative.rmse()):.4f}")
    if hasattr(windowed, "mae"):
        print(f"windowed MAE: {metric_as_float(windowed.mae()):.4f}")
    print(f"windowed RMSE: {metric_as_float(windowed.rmse()):.4f}")

    windowed_mae = metric_as_series(windowed.mae()) if hasattr(windowed, "mae") else []
    windowed_rmse = metric_as_series(windowed.rmse())

    if args.plot:
        x_rmse = window_end_samples(len(windowed_rmse), args.window_size, max_samples or len(windowed_rmse))
        plt.figure(figsize=(10, 5))

        if windowed_mae:
            x_mae = window_end_samples(len(windowed_mae), args.window_size, max_samples or len(windowed_mae))
            plt.plot(x_mae, windowed_mae, label="Windowed MAE", linewidth=2)
        plt.plot(x_rmse, windowed_rmse, label="Windowed RMSE", linewidth=2)

        plt.title(f"Prequential Metrics on Raw Bike (model={args.model})")
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
