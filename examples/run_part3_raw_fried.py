from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from capymoa.datasets import Fried
from capymoa.evaluation import prequential_evaluation
from capymoa.regressor import AdaptiveRandomForestRegressor, FIMTDD, SGDRegressor

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ExperimentHelper


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
    parser = ExperimentHelper.create_prequential_parser(
        description="Prequential evaluation on the raw Fried stream (no forecasting transform).",
        include_forecasting_args=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = ExperimentHelper.normalize_max_samples(args.max_samples)

    stream = Fried()
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
        progress_bar=True,
    )

    cumulative = results["cumulative"]
    windowed = results["windowed"]

    paths = ExperimentHelper.build_output_paths(
        dataset_tag="fried",
        mode_tag="raw",
        model=args.model,
        lag_size=None,
        horizon=None,
        include_input_lags=None,
        window_size=args.window_size,
        max_samples=max_samples,
    )

    cumulative_mae = (
        ExperimentHelper.metric_as_float(cumulative.mae()) if hasattr(cumulative, "mae") else float("nan")
    )
    cumulative_rmse = ExperimentHelper.metric_as_float(cumulative.rmse())
    windowed_mae_final = (
        ExperimentHelper.metric_as_float(windowed.mae()) if hasattr(windowed, "mae") else float("nan")
    )
    windowed_rmse_final = ExperimentHelper.metric_as_float(windowed.rmse())

    print("\n=== Prequential Evaluation (Raw Fried) ===")
    print(f"model: {args.model}")
    print(f"max samples: {max_samples if max_samples is not None else 'all'}")
    print(f"window size: {args.window_size}")
    if hasattr(cumulative, "mae"):
        print(f"cumulative MAE: {cumulative_mae:.4f}")
    print(f"cumulative RMSE: {cumulative_rmse:.4f}")
    if hasattr(windowed, "mae"):
        print(f"windowed MAE: {windowed_mae_final:.4f}")
    print(f"windowed RMSE: {windowed_rmse_final:.4f}")

    windowed_mae = ExperimentHelper.metric_as_series(windowed.mae()) if hasattr(windowed, "mae") else []
    windowed_rmse = ExperimentHelper.metric_as_series(windowed.rmse())
    total_eval = max_samples if max_samples is not None else max(len(windowed_rmse), 1) * args.window_size
    x_rmse = ExperimentHelper.window_end_samples(len(windowed_rmse), args.window_size, total_eval)

    summary = {
        "dataset": "fried",
        "task": "raw",
        "mode": "raw regression",
        "model": args.model,
        "window_size": args.window_size,
        "evaluated_samples": int(x_rmse[-1]) if len(x_rmse) > 0 else 0,
        "cumulative_mae": cumulative_mae,
        "cumulative_rmse": cumulative_rmse,
        "windowed_mae": windowed_mae_final,
        "windowed_rmse": windowed_rmse_final,
    }

    if args.plot:
        plt.figure(figsize=(11, 6))
        ax = plt.gca()

        if windowed_mae:
            x_mae = ExperimentHelper.window_end_samples(len(windowed_mae), args.window_size, total_eval)
            plt.plot(x_mae, windowed_mae, label="Windowed MAE", linewidth=2)
        plt.plot(x_rmse, windowed_rmse, label="Windowed RMSE", linewidth=2)

        plt.title(f"Prequential Metrics on Raw Fried (model={args.model})")
        plt.xlabel("Number of evaluated samples")
        plt.ylabel("Metric value (MAE / RMSE)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        ExperimentHelper.add_summary_box(ax, summary)
        plt.tight_layout()

        plt.savefig(paths["plot"], dpi=140)
        print(f"saved plot: {paths['plot']}")

        if args.show_plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    main()