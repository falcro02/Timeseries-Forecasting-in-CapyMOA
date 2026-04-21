"""Lag-based feature extraction for streaming forecasting problems.

The transformer stores a sliding window of past target values and, optionally,
past input vectors and calendar features. Each call to `step()` consumes one
new observation and emits a supervised sample once enough history exists.
"""

from collections import deque
from datetime import date, datetime
from typing import Iterable, Optional, Sequence


class LagTransformer:
    """Transform a stream of observations into lag-based forecasting features.

    The transformer keeps the last `k` target values in a fixed-size window.
    When `include_input_lags=True`, it also stores the last `k` input vectors.
    When `include_time_features=True`, it appends calendar-derived features.
    """

    def __init__(
        self,
        k: int,
        include_input_lags: bool = False,
        include_time_features: bool = False,
        holiday_dates: Optional[Iterable[date]] = None,
    ):
        """Configure the lag window and optional feature groups to include.

        Args:
            k: Number of past observations to keep in the lag window.
            include_input_lags: If `True`, append past input vectors to features.
            include_time_features: If `True`, append calendar features.
            holiday_dates: Optional collection of dates marked as holidays.
        """
        self.k = k
        self.include_input_lags = include_input_lags
        self.include_time_features = include_time_features
        self.holiday_dates = set(holiday_dates or [])

        self.past_target_window = deque(maxlen=k)
        self.past_feature_window = deque(maxlen=k) if include_input_lags else None

    def _season(self, month: int) -> int:
        """Map a month number to a season code used as a time feature.

        The returned codes are:
        - 0 = winter
        - 1 = spring
        - 2 = summer
        - 3 = autumn
        """
        if not (1 <= month <= 12):
            raise ValueError(
                f"Warning: month must be a number between 1 and 12. Value received: {month}"
            )

        if month in (12, 1, 2):
            return 0  # winter
        if month in (3, 4, 5):
            return 1  # spring
        if month in (6, 7, 8):
            return 2  # summer
        return 3  # autumn

    def _extract_time_features(self, timestamp: datetime) -> list:
        """Extract calendar features from a date or datetime object.

        The returned vector contains:
        hour, weekday, day of month, month, season code, holiday flag.
        """
        if isinstance(timestamp, datetime):
            hour = timestamp.hour
            current_date = timestamp.date()
        elif isinstance(timestamp, date):
            hour = 0
            current_date = timestamp
        else:
            raise TypeError(f"Expected datetime or date. Received: {type(timestamp)}")

        return [
            hour,
            timestamp.weekday(),
            timestamp.day,
            timestamp.month,
            self._season(timestamp.month),
            int(current_date in self.holiday_dates),
        ]

    def step(
        self,
        current_y: float,
        current_x: Optional[Sequence[float]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """Consume one observation and return a forecasting sample when ready.

        The first `k` calls are warm-up calls used to fill the lag window, so the
        method returns `None` until enough history is available.

        Args:
            current_y: Target value of the current stream observation.
            current_x: Input feature vector of the current observation.
            timestamp: Optional timestamp used to derive calendar features.

        Returns:
            A tuple `(features, target)` once the lag window is full, otherwise
            `None`.
        """
        if len(self.past_target_window) < self.k:
            self.past_target_window.append(current_y)
            if self.past_feature_window is not None and current_x is not None:
                self.past_feature_window.append(list(current_x))
            return None

        features = list(self.past_target_window)

        if self.past_feature_window is not None:
            if current_x is None:
                raise ValueError("current_x must be provided when include_input_lags=True")
            if len(self.past_feature_window) < self.k:
                raise ValueError("Not enough input history to build lagged input features")
            for past_x in self.past_feature_window:
                features.extend(list(past_x))

        if self.include_time_features:
            if timestamp is None:
                raise ValueError("timestamp must be provided when include_time_features=True")
            features.extend(self._extract_time_features(timestamp))

        target = current_y

        self.past_target_window.append(current_y)
        if self.past_feature_window is not None and current_x is not None:
            self.past_feature_window.append(list(current_x))

        return features, target
