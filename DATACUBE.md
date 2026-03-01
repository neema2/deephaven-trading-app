# Datacube Engine & UI

A **Legend DataCube–inspired** analytical pivot engine with DuckDB SQL pushdown and a Perspective-based interactive grid. All grouping, pivoting, filtering, and aggregation runs server-side via DuckDB — only the result rows are sent to the browser.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Python Process                       │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Datacube    │───▶│  DuckDB SQL  │───▶│   PyArrow    │  │
│  │   Engine      │    │  Pushdown    │    │   Result     │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                  │          │
│  ┌──────────────┐    ┌──────────────┐           │          │
│  │  Perspective  │◀───│  Arrow IPC   │◀──────────┘          │
│  │  Server (C++) │    │  Serialize   │                      │
│  └──────┬───────┘    └──────────────┘                      │
│         │                                                   │
│  ┌──────┴───────┐    ┌──────────────┐                      │
│  │   Tornado     │    │   /cmd WS    │                      │
│  │  /websocket   │    │  (JSON cmds) │                      │
│  └──────┬───────┘    └──────┬───────┘                      │
└─────────┼───────────────────┼──────────────────────────────┘
          │ Arrow IPC         │ JSON
          ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      Browser                                │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              <perspective-viewer>                  │      │
│  │         Flat grid + row-click drill-down          │      │
│  └──────────────────────────────────────────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │                  Sidebar                           │      │
│  │  Group By · Pivot By · Aggregation · Filters      │      │
│  │  Sort · Drill Path · SQL Preview                  │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
import duckdb
from datacube import Datacube

conn = duckdb.connect()
conn.execute("CREATE TABLE trades AS SELECT * FROM read_parquet('trades.parquet')")

dc = Datacube(conn, source_name="trades")
dc.show()  # opens browser at http://localhost:8050
```

### With Lakehouse

```python
from lakehouse import Lakehouse

lh = Lakehouse()
dc = lh.datacube("trades")  # queries Iceberg via DuckDB
dc.show()
```

### NYC Taxi Demo (3M rows)

```bash
PYTHONPATH=. python3 demos/demo_datacube_ui.py
```

Loads 2.9M taxi rides from parquet over HTTPS. Group by payment type, vendor, zone — all instant via DuckDB pushdown.

## Engine API

The `Datacube` class wraps an immutable `DatacubeSnapshot`. Every mutation returns a **new** instance.

### Sources

| Source | Example |
|--------|---------|
| DuckDB connection + table name | `Datacube(conn, source_name="trades")` |
| Lakehouse table | `lh.datacube("trades")` |
| PyArrow Table | `Datacube(arrow_table, source_name="data")` |
| pandas DataFrame | `Datacube(df, source_name="data")` |
| Raw SQL | `Datacube(conn, source_name="(SELECT * FROM t WHERE x > 0)")` |

### Grouping & Pivoting

```python
dc = dc.set_group_by("sector", "symbol")   # VPivot — GROUP BY dimensions
dc = dc.set_pivot_by("side")               # HPivot — column headers from values
dc = dc.set_group_by()                     # clear group by
dc = dc.set_pivot_by()                     # clear pivot
```

### Per-Column Configuration

```python
dc = dc.set_column("price", aggregate_operator="avg")     # change aggregation
dc = dc.set_column("symbol", is_selected=False)           # hide column
dc = dc.set_column("sector", excluded_from_pivot=True)    # exclude from HPivot
```

Aggregate operators: `sum`, `avg`, `count`, `min`, `max`, `first`, `last`.

### Filtering

```python
dc = dc.add_filter("price", "gt", 100)
dc = dc.add_filter("sector", "eq", "Tech")
dc = dc.add_filter("name", "contains", "apple")
dc = dc.clear_filters()
```

Operators: `eq`, `ne`, `gt`, `lt`, `ge`, `le`, `like`, `contains`, `is_null`, `is_not_null`.

### Sorting

```python
dc = dc.set_sort(("price", True), ("symbol", False))  # (field, descending)
```

### Drill-Down

```python
dc = dc.drill_down(sector="Tech")      # filter + advance depth
dc = dc.drill_down(symbol="AAPL")      # deeper
dc = dc.drill_up()                      # back one level
dc = dc.drill_reset()                   # back to top
```

### Extended Columns

```python
# Leaf-level (pre-aggregation) — computed per raw row
dc = dc.add_leaf_extend("notional", "price * quantity", type="float")

# Group-level (post-aggregation) — computed on grouped results
dc = dc.add_group_extend("avg_price", "price / count", type="float")
```

### Joins

```python
dc = dc.add_join("sectors", on={"symbol": "symbol"}, join_type="LEFT")
```

### Query Execution

```python
arrow = dc.query()           # → PyArrow Table
df = dc.query_df()           # → pandas DataFrame
rows = dc.query_dicts()      # → list[dict]
sql = dc.sql()               # → compiled SQL string
cols = dc.result_columns()   # → column names in result
```

### Serialization

```python
json_str = dc.to_json()      # snapshot → JSON (for persistence/sharing)
```

## UI Features

### Interactive Grid (`dc.show()`)

Opens a Tornado server in-process (like `plt.show()`) with a Perspective flat grid.

| Feature | How it works |
|---------|-------------|
| **Group By** | Sidebar checkboxes → engine `GROUP BY` → DuckDB pushdown |
| **Tree Expand** | Click ▸ group row → children appear inline (▾ to collapse) |
| **Pivot By** | Sidebar checkboxes → engine `PIVOT` → new column headers |
| **Aggregation** | Dropdown per measure → `sum`, `avg`, `count`, etc. |
| **Filters** | Add/clear from sidebar → engine `WHERE` clause |
| **Sort** | Add sort rules from sidebar |
| **SQL Preview** | Collapsible panel showing compiled DuckDB SQL |
| **Row Count** | Status bar shows result size |

### Versioned Tables

Perspective tables have fixed schemas. When group-by or pivot changes the result columns, the server creates a new versioned table (`dc_0`, `dc_1`, `dc_2`...) and tells the client to reconnect. Same-schema updates use fast in-place `replace()`.

### Tree Expand/Collapse

When Group By is active, rows show as collapsible groups:

```
__tree__           sector   symbol  price    quantity
▸ Tech             Tech     —       690.0    2200.0
▸ Finance          Finance  —       530.0    1100.0
▸ Health           Health   —       198.0    1800.0
```

Click ▸ Tech to expand:

```
__tree__           sector   symbol  price    quantity
▾ Tech             Tech     —       690.0    2200.0
    AAPL           Tech     AAPL    150.0    1000.0
    GOOG           Tech     GOOG    140.0    500.0
    MSFT           Tech     MSFT    400.0    700.0
▸ Finance          Finance  —       530.0    1100.0
▸ Health           Health   —       198.0    1800.0
```

Parent rows are aggregated (depth 0). Children are raw leaf rows (depth 1, limited to 200 for responsiveness). Click again to collapse.

## SQL Pushdown

The engine compiles all operations to a single DuckDB SQL query:

```sql
-- Group by sector with sum aggregation
SELECT sector, SUM(price) AS price, SUM(quantity) AS quantity
FROM trades
WHERE price > 100
GROUP BY sector
ORDER BY sector
```

```sql
-- HPivot: side values become column headers
SELECT sector,
  SUM(CASE WHEN side = 'BUY' THEN price END) AS "BUY__|__price",
  SUM(CASE WHEN side = 'SELL' THEN price END) AS "SELL__|__price"
FROM trades
GROUP BY sector
```

No data enters Python memory for the query — DuckDB reads directly from Parquet/Iceberg files.

## Snapshot Model

The `DatacubeSnapshot` is an immutable frozen dataclass:

```python
@dataclass(frozen=True)
class DatacubeSnapshot:
    source: str                                    # table name or SQL
    columns: tuple[DatacubeColumnConfig, ...]      # per-column config
    group_by: tuple[str, ...] = ()                 # VPivot dimensions
    pivot_by: tuple[str, ...] = ()                 # HPivot dimensions
    filters: tuple[Filter, ...] = ()               # WHERE predicates
    sort: tuple[Sort, ...] = ()                    # ORDER BY
    drill_path: tuple[dict, ...] = ()              # drill-down breadcrumb
    leaf_extended_columns: tuple[ExtendedColumn, ...] = ()
    group_extended_columns: tuple[ExtendedColumn, ...] = ()
    joins: tuple[JoinSpec, ...] = ()               # table joins
    limit: int | None = None
    offset: int | None = None
    pivot_values: tuple[str, ...] | None = None    # cached pivot value discovery
    pivot_statistic_column: str | None = None       # "Total" column
```

Every user interaction creates a new snapshot → recompiles SQL → re-queries DuckDB.

## Files

```
datacube/
├── __init__.py         # package init
├── config.py           # DatacubeSnapshot, DatacubeColumnConfig, Filter, Sort, etc.
├── compiler.py         # Snapshot → DuckDB SQL compiler
├── engine.py           # Datacube class (user-facing API + show())
├── server.py           # Tornado + Perspective server + tree builder
└── static/
    └── index.html      # Perspective viewer + sidebar controls
```

## Dependencies

```
perspective-python     # FINOS Perspective (C++ engine + Tornado handler)
tornado                # WebSocket server
duckdb                 # SQL pushdown engine
pyarrow                # Arrow IPC serialization
```

## Design Decisions

1. **DuckDB pushdown, not client-side grouping** — with billion-row tables, the engine runs `GROUP BY sector` via DuckDB and returns 5 rows, not 1B rows to the browser.

2. **Perspective as flat grid renderer** — Perspective's native `group_by` requires loading all data into memory. Instead, we use it as a high-performance flat grid and handle grouping/pivoting server-side.

3. **Immutable snapshots** — every mutation returns a new `Datacube`. No shared mutable state. Snapshots are serializable to JSON for persistence.

4. **Versioned tables for schema changes** — Perspective tables can't change schema after creation. The server creates new versioned tables when group-by or pivot changes the column set.

5. **Arrow-native tree builder** — the `_build_tree_result()` function uses Arrow slice/concat operations (not Python dicts) to preserve types like datetimes correctly.

6. **Inspired by FINOS Legend DataCube** — snapshot-based state machine, per-column configuration, VPivot (group_by) + HPivot (pivot_by), leaf/group extended columns, drill-down with breadcrumb.

## Test Coverage

- **72 unit tests** (`test_datacube.py`) — snapshot model, SQL compiler, column config, pivot, drill-down, joins, extended columns
- **40 integration tests** (`test_datacube_integration.py`) — end-to-end with real DuckDB, Lakehouse detection, query execution
