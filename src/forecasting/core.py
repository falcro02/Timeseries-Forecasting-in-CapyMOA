"""Core utilities for turning CapyMOA streams into forecasting datasets.

This module keeps the public pipeline small on purpose:
- transform one stream observation into lag-based features;
- optionally aggregate one-step targets into an H-step target;
- return plain Python lists that can be used by a model or exported.
"""

import numpy as np
from capymoa.stream import Stream, Schema
from capymoa.instance import RegressionInstance
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


class ForecastingStream(Stream):
    """A CapyMOA Stream that processes an upstream time series in a purely streaming fashion.
    
    It consumes instances from the source stream, transforms them via LagTransformer, 
    aggregates targets via HorizonAggregator, and yields them sequentially.
    """

    def __init__(self, source_stream: Stream, transformer: LagTransformer, horizon: int = 1, max_samples: Optional[int] = None):
        """Create a streaming dataset that evaluates targets H steps ahead.
        
        Args:
            source_stream: The underlying CapyMOA stream (e.g. Bike() or Fried()).
            transformer: The tool that lags observations into a feature vector.
            horizon: Number of future targets to average (or 1 for one-step delay).
            max_samples: Optional limit on the number of emitted samples.
        """
        self.source_stream = source_stream
        self.transformer = transformer
        self.horizon = horizon
        self.max_samples = max_samples
        self.aggregator = HorizonAggregator(horizon=horizon)
        
        self.emitted_count = 0
        self._next_sample = None
        self._schema = None

        # Fetch the very first sample to construct the schema before evaluation starts
        self._poll_next()
        if self._next_sample is not None:
            num_features = len(self._next_sample.features)
            feature_names = [f"lag_feature_{i}" for i in range(num_features)]
            # Schema.from_custom expects the target attribute to appear in the
            # features list (example in capymoa docs). Append the target name.
            schema_features = feature_names + ["target"]
            self._schema = Schema.from_custom(
                features=schema_features,
                target="target",
                name="ForecastingTransformed",
            )

    def _poll_next(self):
        """Consume source instances until a valid sample is available from the aggregator."""
        if self.max_samples is not None and self.emitted_count >= self.max_samples:
            self._next_sample = None
            return

        while self.source_stream.has_more_instances():
            instance = self.source_stream.next_instance()
            
            y_val = float(instance.y_value)
            x_val = getattr(instance, "x", None)
            timestamp = getattr(instance, "timestamp", None)
            
            result = self.transformer.step(
                current_y=y_val, 
                current_x=x_val, 
                timestamp=timestamp
            )
            
            if result is None:
                continue
                
            features, target = result
            aggregated = self.aggregator.step(features, target)
            
            if aggregated is not None:
                self._next_sample = aggregated
                return
                
        self._next_sample = None

    def has_more_instances(self) -> bool:
        return self._next_sample is not None

    def next_instance(self) -> RegressionInstance:
        if self._next_sample is None:
            raise StopIteration("No more transformed instances available.")
            
        if self._schema is None:
            raise ValueError("Schema is not initialized yet.")

        instance_features = np.array(self._next_sample.features, dtype=np.float64)
        instance_target = np.float64(self._next_sample.target)
        
        # Build CapyMOA RegressionInstance directly
        inst = RegressionInstance.from_array(self._schema, instance_features, instance_target)
        
        self.emitted_count += 1
        self._poll_next()
        return inst

    def get_schema(self) -> Schema:
        if self._schema is None:
            raise ValueError("Schema is not initialized yet.")
        return self._schema

    def restart(self) -> None:
        """Reset internal states and restart the source stream."""
        self.source_stream.restart()
        # Since transformer has no reset, we assume the user instantiates a new one
        # or we just clear its internal queue manually.
        if hasattr(self.transformer, 'past_target_window'):
            self.transformer.past_target_window.clear()
        if hasattr(self.transformer, 'past_feature_window') and self.transformer.past_feature_window is not None:
            self.transformer.past_feature_window.clear()
        self.aggregator = HorizonAggregator(horizon=self.horizon)
        self.emitted_count = 0
        self._next_sample = None
        # Preserve schema or regenerate? Regenerate is safer.
        self._schema = None
        self._poll_next()
        if self._next_sample is not None:
            num_features = len(self._next_sample.features)
            feature_names = [f"lag_feature_{i}" for i in range(num_features)]
            # Schema.from_custom expects the target attribute to appear in the
            # features list (example in capymoa docs). Append the target name.
            schema_features = feature_names + ["target"]
            self._schema = Schema.from_custom(
                features=schema_features,
                target="target",
                name="ForecastingTransformed",
            )
