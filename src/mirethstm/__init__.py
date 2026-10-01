"""MirethSTM1: a free decision engine that scores typed answers with a local model."""

from .engine import Engine
from .schema import SchemaError

__version__ = "0.0.1"

__all__ = ["Engine", "SchemaError", "__version__"]
