from dataclasses import dataclass

@dataclass
class Metric:
    value: float
    unit: str

@dataclass
class ThresholdMetrics:
    area_under_threshold: Metric
    minutes_under_threshold: Metric
