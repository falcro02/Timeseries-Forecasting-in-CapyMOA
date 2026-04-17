from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Iterable, Optional, Sequence

from .lag_transformer import LagTransformer


@dataclass
class ForecastSample:
    features: list[float]
    target: float


@dataclass
class PendingForecast:
    features: list[float]
    future_targets: list[float]


class HorizonAggregator:
    """Aggregates one-step targets into a horizon target of size H."""

    def __init__(self, horizon: int):
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        self.horizon = horizon
        self.pending: Deque[PendingForecast] = deque()

    def step(self, features: Sequence[float], target_t1: float) -> Optional[ForecastSample]:
        for item in self.pending:
            item.future_targets.append(float(target_t1))

        self.pending.append(PendingForecast(list(features), [float(target_t1)]))

        if self.pending and len(self.pending[0].future_targets) >= self.horizon:
            ready = self.pending.popleft()
            return ForecastSample(
                features=ready.features,
                target=float(mean(ready.future_targets[: self.horizon])),
            )

        return None


class ForecastDatasetBuilder:
    """Converts CapyMOA regression stream instances into forecasting samples.

    Expected instance fields:
    - y_value: regression target
    - x: input feature vector
    Optional field:
    - timestamp: used only if LagTransformer.include_time_features=True
    """

    def __init__(self, transformer: LagTransformer):
        self.transformer = transformer

    def _transform_step(self, instance) -> Optional[ForecastSample]:
        current_y = float(instance.y_value)
        current_x: Optional[Sequence[float]] = instance.x
        timestamp = getattr(instance, "timestamp", None)

        result = self.transformer.step(
            current_y=current_y,
            current_x=current_x,
            timestamp=timestamp,
        )
        if result is None:
            return None

        features, target = result
        return ForecastSample(features=list(features), target=float(target))

    def build_one_step(
        self,
        source: Iterable,
        max_samples: Optional[int] = None,
    ) -> list[ForecastSample]:
        samples: list[ForecastSample] = []

        for instance in source:
            transformed = self._transform_step(instance)
            if transformed is None:
                continue

            samples.append(transformed)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples

    def build_aggregated_horizon(
        self,
        source: Iterable,
        horizon: int,
        max_samples: Optional[int] = None,
    ) -> list[ForecastSample]:
        aggregator = HorizonAggregator(horizon=horizon)
        samples: list[ForecastSample] = []

        for instance in source:
            transformed = self._transform_step(instance)
            if transformed is None:
                continue

            aggregated = aggregator.step(transformed.features, transformed.target)
            if aggregated is None:
                continue

            samples.append(aggregated)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples
