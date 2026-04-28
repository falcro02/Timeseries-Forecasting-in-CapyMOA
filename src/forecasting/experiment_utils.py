from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class ExperimentHelper:
    """Shared CLI, output, and metric utilities for experiment scripts."""

    @staticmethod
    def create_prequential_parser(
        description: str,
        include_forecasting_args: bool,
        default_max_samples: int,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)

        if include_forecasting_args:
            parser.add_argument("--lag-size", type=int, default=1, help="Lag window size k.")
            parser.add_argument("--horizon", type=int, default=1, help="Forecasting horizon H.")
            parser.add_argument(
                "--include-input-lags",
                action="store_true",
                help="Include lagged input features in addition to lagged target.",
            )

        parser.add_argument(
            "--max-samples",
            type=int,
            default=default_max_samples,
            help="Maximum number of instances to evaluate. Use -1 for all.",
        )
        parser.add_argument(
            "--model",
            choices=["sgd", "arf", "fimtdd"],
            default="arf",
            help="CapyMOA regressor used in prequential evaluation.",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.5e-5,
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
            help="Create MAE/RMSE windowed plot (default: true).",
        )
        parser.add_argument(
            "--show-plot",
            action="store_true",
            help="Display the plot window interactively.",
        )

        return parser

    @staticmethod
    def normalize_max_samples(max_samples: int) -> int | None:
        return None if max_samples < 0 else max_samples

    @staticmethod
    def metric_as_float(value: Any) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return float("nan")
            return float(value[-1])
        return float(value)

    @staticmethod
    def metric_as_series(value: Any) -> list[float]:
        if isinstance(value, np.ndarray):
            return [float(v) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return [float(value)]

    @staticmethod
    def window_end_samples(num_points: int, window_size: int, total_samples: int) -> np.ndarray:
        return np.array(
            [min((idx + 1) * window_size, total_samples) for idx in range(num_points)],
            dtype=int,
        )

    @staticmethod
    def build_output_paths(
        dataset_tag: str,
        mode_tag: str,
        model: str,
        lag_size: int | None,
        horizon: int | None,
        include_input_lags: bool | None,
        window_size: int,
        max_samples: int | None,
    ) -> dict[str, Path]:
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        n_tag = "all" if max_samples is None else str(max_samples)
        lag_tag = f"_k{lag_size}" if lag_size is not None else ""
        h_tag = f"_h{horizon}" if horizon is not None else ""
        inlags_tag = ""
        if include_input_lags is not None:
            inlags_tag = f"_inlags{int(include_input_lags)}"

        base_name = (
            f"{dataset_tag}_{mode_tag}_{model}{lag_tag}{h_tag}{inlags_tag}_ws{window_size}_n{n_tag}"
        )

        return {"plot": out_dir / f"{base_name}.png"}

    @staticmethod
    def add_summary_box(ax: Any, summary: dict[str, Any]) -> None:
        lines = [
            f"samples: {summary['evaluated_samples']}",
            f"cum MAE: {summary['cumulative_mae']:.3f}",
            f"cum RMSE: {summary['cumulative_rmse']:.3f}",
            f"win MAE: {summary['windowed_mae']:.3f}",
            f"win RMSE: {summary['windowed_rmse']:.3f}",
        ]
        ax.text(
            0.99,
            0.01,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
        )
