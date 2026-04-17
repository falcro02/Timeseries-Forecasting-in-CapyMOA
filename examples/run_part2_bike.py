from pathlib import Path
import sys
from typing import Any, Sequence

from capymoa.datasets import Bike

# Make src/ importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting import ForecastDatasetBuilder, LagTransformer


def bike_target(instance: Any) -> float:
    return float(instance.y_value)


def bike_features(instance: Any) -> Sequence[float]:
    return instance.x


def main() -> None:
    lag_size = 24
    horizon = 6

    stream_one_step = Bike()
    one_step_builder = ForecastDatasetBuilder(
        transformer=LagTransformer(k=lag_size, include_input_lags=True),
        get_target=bike_target,
        get_features=bike_features,
    )
    one_step_samples = one_step_builder.build_one_step(
        source=stream_one_step,
        max_samples=2000,
    )

    stream_agg = Bike()
    agg_builder = ForecastDatasetBuilder(
        transformer=LagTransformer(k=lag_size, include_input_lags=True),
        get_target=bike_target,
        get_features=bike_features,
    )
    aggregated_samples = agg_builder.build_aggregated_horizon(
        source=stream_agg,
        horizon=horizon,
        max_samples=2000,
    )

    print(f"One-step samples: {len(one_step_samples)}")
    if one_step_samples:
        first = one_step_samples[0]
        print(f"One-step first sample -> x_len={len(first.features)}, y={first.target:.3f}")

    print(f"Aggregated H={horizon} samples: {len(aggregated_samples)}")
    if aggregated_samples:
        first = aggregated_samples[0]
        print(
            f"Aggregated first sample -> x_len={len(first.features)}, y_mean={first.target:.3f}"
        )


if __name__ == "__main__":
    main()
