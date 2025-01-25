"""Common type aliases used in the package."""

from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Any, Union, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from vitabel import Channel, Label

Timedelta: TypeAlias = pd.Timedelta | np.timedelta64
"""Type alias of a time difference / duration."""

Timestamp: TypeAlias = pd.Timestamp | np.datetime64
"""Type alias of a time stamp."""

ChannelSpecification: TypeAlias = Union[str, dict[str, Any], "Channel"]
"""Type alias for different ways to specify a Channel."""

LabelSpecification: TypeAlias = Union[str, dict[str, Any], "Label"]
"""Type alias for different ways to specify a Label."""