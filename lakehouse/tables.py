"""
Lakehouse Table Definitions
=============================
Iceberg table schemas for events, ticks, bars, and positions.
All tables created in the 'default' namespace.
"""

from __future__ import annotations

import logging

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    TableAlreadyExistsError,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import DayTransform, IdentityTransform, MonthTransform
from pyiceberg.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    NestedField,
    StringType,
    TimestamptzType,
)

logger = logging.getLogger(__name__)

NAMESPACE = "default"

# ── Schema: events ──────────────────────────────────────────────────────────
# Source: PG object_events table (bi-temporal event log)

EVENTS_SCHEMA = Schema(
    NestedField(1, "event_id", StringType(), required=True),
    NestedField(2, "entity_id", StringType(), required=True),
    NestedField(3, "version", IntegerType(), required=True),
    NestedField(4, "type_name", StringType(), required=True),
    NestedField(5, "owner", StringType(), required=False),
    NestedField(6, "updated_by", StringType(), required=False),
    NestedField(7, "data", StringType(), required=False),       # JSON string
    NestedField(8, "state", StringType(), required=False),
    NestedField(9, "event_type", StringType(), required=True),
    NestedField(10, "event_meta", StringType(), required=False), # JSON string
    NestedField(11, "tx_time", TimestamptzType(), required=True),
    NestedField(12, "valid_from", TimestamptzType(), required=True),
    NestedField(13, "valid_to", TimestamptzType(), required=False),
)

EVENTS_PARTITION = PartitionSpec(
    PartitionField(
        source_id=4, field_id=1000, transform=IdentityTransform(), name="type_name_identity"
    ),
    PartitionField(
        source_id=11, field_id=1001, transform=DayTransform(), name="tx_time_day"
    ),
)

# ── Schema: ticks ───────────────────────────────────────────────────────────
# Source: QuestDB equity_ticks, fx_ticks, curve_ticks (unified)

TICKS_SCHEMA = Schema(
    NestedField(1, "tick_type", StringType(), required=True),    # "equity", "fx", "curve"
    NestedField(2, "symbol", StringType(), required=True),       # symbol/pair/label
    NestedField(3, "price", DoubleType(), required=False),       # equity
    NestedField(4, "bid", DoubleType(), required=False),
    NestedField(5, "ask", DoubleType(), required=False),
    NestedField(6, "mid", DoubleType(), required=False),         # fx
    NestedField(7, "volume", LongType(), required=False),        # equity
    NestedField(8, "change", DoubleType(), required=False),
    NestedField(9, "change_pct", DoubleType(), required=False),
    NestedField(10, "spread_pips", DoubleType(), required=False), # fx
    NestedField(11, "rate", DoubleType(), required=False),        # curve
    NestedField(12, "tenor_years", DoubleType(), required=False), # curve
    NestedField(13, "discount_factor", DoubleType(), required=False),
    NestedField(14, "currency", StringType(), required=False),
    NestedField(15, "timestamp", TimestamptzType(), required=True),
)

TICKS_PARTITION = PartitionSpec(
    PartitionField(
        source_id=1, field_id=1000, transform=IdentityTransform(), name="tick_type_identity"
    ),
    PartitionField(
        source_id=15, field_id=1001, transform=DayTransform(), name="timestamp_day"
    ),
)

# ── Schema: bars_daily ──────────────────────────────────────────────────────
# Source: QuestDB OHLCV bar aggregations

BARS_SCHEMA = Schema(
    NestedField(1, "symbol", StringType(), required=True),
    NestedField(2, "tick_type", StringType(), required=True),
    NestedField(3, "interval", StringType(), required=True),     # "1d"
    NestedField(4, "open", DoubleType(), required=True),
    NestedField(5, "high", DoubleType(), required=True),
    NestedField(6, "low", DoubleType(), required=True),
    NestedField(7, "close", DoubleType(), required=True),
    NestedField(8, "volume", LongType(), required=False),
    NestedField(9, "trade_count", IntegerType(), required=False),
    NestedField(10, "timestamp", TimestamptzType(), required=True),
)

BARS_PARTITION = PartitionSpec(
    PartitionField(
        source_id=1, field_id=1000, transform=IdentityTransform(), name="symbol_identity"
    ),
    PartitionField(
        source_id=10, field_id=1001, transform=MonthTransform(), name="timestamp_month"
    ),
)

# ── Schema: positions ──────────────────────────────────────────────────────
# Source: PG object_events filtered to type_name='Position' (latest version)

POSITIONS_SCHEMA = Schema(
    NestedField(1, "entity_id", StringType(), required=True),
    NestedField(2, "symbol", StringType(), required=True),
    NestedField(3, "quantity", IntegerType(), required=True),
    NestedField(4, "avg_cost", DoubleType(), required=False),
    NestedField(5, "current_price", DoubleType(), required=False),
    NestedField(6, "market_value", DoubleType(), required=False),
    NestedField(7, "unrealized_pnl", DoubleType(), required=False),
    NestedField(8, "side", StringType(), required=False),
    NestedField(9, "state", StringType(), required=False),
    NestedField(10, "owner", StringType(), required=False),
    NestedField(11, "valid_from", TimestamptzType(), required=True),
    NestedField(12, "is_deleted", BooleanType(), required=False),
)

POSITIONS_PARTITION = PartitionSpec(
    PartitionField(
        source_id=11, field_id=1000, transform=DayTransform(), name="valid_from_day"
    ),
)


# ── Table registry ──────────────────────────────────────────────────────────

TABLE_DEFS = {
    "events": (EVENTS_SCHEMA, EVENTS_PARTITION),
    "ticks": (TICKS_SCHEMA, TICKS_PARTITION),
    "bars_daily": (BARS_SCHEMA, BARS_PARTITION),
    "positions": (POSITIONS_SCHEMA, POSITIONS_PARTITION),
}


def ensure_tables(catalog: Catalog, namespace: str = NAMESPACE) -> dict:
    """
    Create all Iceberg tables if they don't exist. Idempotent.

    Returns dict of table_name → Table object.
    """
    # Ensure namespace exists
    try:
        catalog.create_namespace(namespace)
        logger.info("Created namespace '%s'", namespace)
    except NamespaceAlreadyExistsError:
        pass

    tables = {}
    for table_name, (schema, partition_spec) in TABLE_DEFS.items():
        identifier = (namespace, table_name)
        try:
            table = catalog.create_table(
                identifier=identifier,
                schema=schema,
                partition_spec=partition_spec,
            )
            logger.info("Created Iceberg table '%s.%s'", namespace, table_name)
            tables[table_name] = table
        except TableAlreadyExistsError:
            table = catalog.load_table(identifier)
            logger.info("Iceberg table '%s.%s' already exists", namespace, table_name)
            tables[table_name] = table

    return tables
