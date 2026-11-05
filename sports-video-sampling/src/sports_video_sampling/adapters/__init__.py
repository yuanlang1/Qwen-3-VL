"""Normalize dataset formats before invoking the sports sampling core."""

from .qwen import convert_qwen
from .sportsqa import convert_sportsqa

CONVERTERS = {"qwen": convert_qwen, "sportsqa": convert_sportsqa}
