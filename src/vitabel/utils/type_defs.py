from dataclasses import dataclass

@dataclass
class Metric:
    value: float
    unit: str

@dataclass
class ThresholdMetrics:
    area_under_threshold: Metric
    minutes_under_threshold: Metric
    time_weighted_average_under_threshold: Metric
    minutes_observational_interval: Metric
