"""
Example domain classes for the object store.
Each is a @dataclass subclassing Storable.
"""

from dataclasses import dataclass

from store.base import Storable


@dataclass
class Trade(Storable):
    """A single trade execution."""
    symbol: str = ""
    quantity: int = 0
    price: float = 0.0
    side: str = ""  # "BUY" or "SELL"
    timestamp: str | None = None


@dataclass
class Order(Storable):
    """A pending or filled order."""
    symbol: str = ""
    quantity: int = 0
    price: float = 0.0
    side: str = ""
    order_type: str = "LIMIT"  # "LIMIT", "MARKET", "STOP"
    status: str = "PENDING"    # "PENDING", "FILLED", "CANCELLED"
    timestamp: str | None = None


@dataclass
class Signal(Storable):
    """A quant trading signal."""
    symbol: str = ""
    direction: str = ""   # "LONG" or "SHORT"
    strength: float = 0.0  # 0.0 to 1.0
    model_name: str = ""
    notes: str = ""
    timestamp: str | None = None
