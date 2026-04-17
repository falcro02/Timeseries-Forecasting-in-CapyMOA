from collections import deque
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Callable, Deque, Generic, Iterable, Optional, Sequence, TypeVar

from .lag_transformer import LagTransformer

Observation = TypeVar("Observation")
TargetExtractor = Callable[[Observation], float]
FeatureExtractor = Callable[[Observation], Optional[Sequence[float]]]
TimestampExtractor = Callable[[Observation], Optional[datetime]]


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


class ForecastDatasetBuilder(Generic[Observation]):
    """Converts any iterable data source into forecasting samples."""

    def __init__(
        self,
        transformer: LagTransformer,
        get_target: TargetExtractor[Observation],
        get_features: Optional[FeatureExtractor[Observation]] = None,
        get_timestamp: Optional[TimestampExtractor[Observation]] = None,
    ):
        self.transformer = transformer
        self.get_target = get_target
        self.get_features = get_features
        self.get_timestamp = get_timestamp

    def _transform_step(self, observation: Observation) -> Optional[ForecastSample]:
        current_y = float(self.get_target(observation))
        current_x = self.get_features(observation) if self.get_features is not None else None
        timestamp = self.get_timestamp(observation) if self.get_timestamp is not None else None

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
        source: Iterable[Observation],
        max_samples: Optional[int] = None,
    ) -> list[ForecastSample]:
        samples: list[ForecastSample] = []

        for observation in source:
            transformed = self._transform_step(observation)
            if transformed is None:
                continue

            samples.append(transformed)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples

    def build_aggregated_horizon(
        self,
        source: Iterable[Observation],
        horizon: int,
        max_samples: Optional[int] = None,
    ) -> list[ForecastSample]:
        aggregator = HorizonAggregator(horizon=horizon)
        samples: list[ForecastSample] = []

        for observation in source:
            transformed = self._transform_step(observation)
            if transformed is None:
                continue

            aggregated = aggregator.step(transformed.features, transformed.target)
            if aggregated is None:
                continue

            samples.append(aggregated)

            if max_samples is not None and len(samples) >= max_samples:
                break

        return samples
