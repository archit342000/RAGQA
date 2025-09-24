"""Serialize thread units with strict order guarantees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from ..threading.threader import ThreadUnit


@dataclass(slots=True)
class SerializationLog:
    """Captures telemetry emitted during serialization."""

    order_keys: List[tuple[str, int, int, int, int]] = field(default_factory=list)


class Serializer:
    """Ensure strictly monotonic :class:`ThreadUnit` order."""

    def __init__(self):
        self.log = SerializationLog()
        self._last_key: tuple[str, int, int, int, int] | None = None

    def serialize(self, units: Sequence[ThreadUnit]) -> List[ThreadUnit]:
        ordered: List[ThreadUnit] = []
        for unit in units:
            key = unit.order_key
            if self._last_key is not None and key < self._last_key:
                raise ValueError(f"Order regression detected: {key} < {self._last_key}")
            ordered.append(unit)
            self._last_key = key
            self.log.order_keys.append(key)
        return ordered

