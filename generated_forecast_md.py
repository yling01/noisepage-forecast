from dataclasses import dataclass

import pandas as pd


@dataclass
class GeneratedForecastMD:
    target_timestamp: str
    granularity: pd.Timedelta
