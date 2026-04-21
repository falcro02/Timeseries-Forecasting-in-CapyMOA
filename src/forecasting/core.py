"""Core utilities for turning CapyMOA streams into forecasting datasets.

This module keeps the public pipeline small on purpose:
- transform one stream observation into lag-based features;
- optionally aggregate one-step targets into an H-step target;
- return plain Python lists that can be used by a model or exported.
"""

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Iterable, Optional, Sequence

from .lag_transformer import LagTransformer


@dataclass
class ForecastSample:
    """A single supervised forecasting example.

    features contains the lagged inputs produced by the transformer.
    target contains either the one-step value or the aggregated horizon value.
    """

    features: list[float]
    target: float


@dataclass
class PendingForecast:
    """Internal state for one sample waiting for future targets.

    future_targets stores the values collected after the sample was created.
    Once enough values are available, the sample can be emitted.
    """

    features: list[float]
    future_targets: list[float]


class HorizonAggregator:
    """Aggregate one-step targets into a horizon target of size H.

    The first transformed sample is kept in a queue until H future targets are
    available. At that point, the returned target is the mean of those values.
    With H=1, this behaves like standard one-step-ahead forecasting.
    """

    def __init__(self, horizon: int):
        """Create an aggregator for a positive horizon.

        Args:
            horizon: Number of future one-step targets to average.

        Raises:
            ValueError: If horizon is less than or equal to zero.
        """
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        self.horizon = horizon
        self.pending: Deque[PendingForecast] = deque()

    def step(self, features: Sequence[float], target_t1: float) -> Optional[ForecastSample]:
        """Push one one-step sample and emit the oldest ready horizon sample.

        Args:
            features: Lagged feature vector for the current observation.
            target_t1: The next one-step target observed in the stream.

        Returns:
            A ForecastSample when the oldest pending sample has collected
            enough future targets, otherwise None.
        """
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
    """Convert CapyMOA stream instances into forecasting samples.

    Expected stream instance attributes:
    - y_value: numeric regression target.
    - x: input feature vector.
    - timestamp (optional): used only when the transformer requests time features.
    """

    def __init__(self, transformer: LagTransformer):
        """Store the lag transformer used to convert stream observations.

        Args:
            transformer: The lag transformer that produces forecasting features.
        """
        self.transformer = transformer

    def _transform_step(self, instance) -> Optional[ForecastSample]:
        """Transform one stream instance into a one-step forecasting sample.

        The method returns None during the warm-up phase, when the transformer
        does not yet have enough history to build a lag window.
        """
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

    def build_forecasting_dataset(
        self,
        source: Iterable,
        horizon: int,
        max_samples: Optional[int] = None,
    ) -> list[ForecastSample]:
        """Build forecasting samples whose target is the mean over the next H one-step targets.

        Args:
            source: Iterable of stream instances.
            horizon: Number of future one-step targets to average.
            max_samples: Optional cap on the number of returned samples.

        Returns:
            A list of forecasting samples ready to be used by a regressor.
        """
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
    