from collections import deque
from datetime import date, datetime
from typing import Iterable, Optional, Sequence


class LagTransformer:
    def __init__(
        self,
        k: int,
        include_input_lags: bool = False,
        include_time_features: bool = False,
        holiday_dates: Optional[Iterable[date]] = None,
    ):
        self.k = k
        self.include_input_lags = include_input_lags
        self.include_time_features = include_time_features
        self.holiday_dates = set(holiday_dates or [])

        self.past_target_window = deque(maxlen=k)
        self.past_feature_window = deque(maxlen=k) if include_input_lags else None

    def _season(self, month: int) -> int:
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
        if len(self.past_target_window) < self.k:
            self.past_target_window.append(current_y)
            if self.include_input_lags and current_x is not None:
                if self.past_feature_window is None:
                    self.past_feature_window = deque(maxlen=self.k)
                self.past_feature_window.append(list(current_x))
            return None

        features = list(self.past_target_window)

        if self.include_input_lags:
            if current_x is None:
                raise ValueError("current_x must be provided when include_input_lags=True")
            if self.past_feature_window is None:
                self.past_feature_window = deque(maxlen=self.k)
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
        if self.include_input_lags and current_x is not None:
            if self.past_feature_window is None:
                self.past_feature_window = deque(maxlen=self.k)
            self.past_feature_window.append(list(current_x))

        return features, target
