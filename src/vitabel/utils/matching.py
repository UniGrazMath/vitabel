"""Matching objects against filter criteria."""

from __future__ import annotations

__all__ = ["match_object"]


def match_object(obj: object, **kwargs) -> bool:
    """Check whether a given object matches the filter criteria.

    Parameters
    ----------
    obj
        The object to check.
    kwargs
        Filter criteria. The keys are the attribute names of the object,
        and the values are the expected values. If expected value is a
        dictionary, then the object attribute only needs to contain all
        key-value pairs in the dictionary (but could contain other key-value
        pairs too).
    """
    for key, value in kwargs.items():
        if isinstance(value, dict):
            if not all(getattr(obj, key).get(k) == v for k, v in value.items()):
                return False
        else:
            if not hasattr(obj, key):
                return False
            if getattr(obj, key) != value:
                return False
    return True
