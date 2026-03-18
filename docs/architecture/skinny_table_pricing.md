# Skinny Table Swap Pricing Architecture

## Motivation
Historically, `py-flow` generated full, unrolled Abstract Syntax Trees (ASTs) for every distinct instrument in the portfolio. While robust, this approach encounters scaling limits when pricing portfolios of thousands of swaps across large scenario sets. The overhead of generating, transmitting, and compiling the analytic ASTs for every individual swap's cashflows becomes computationally prohibitive and creates large payloads.

## The Skinny Table Approach
The Skinny Table (or Narrow Schema) architecture solves this by decoupling the complex non-linear curve math (interpolations, root finding, OIS projection dependencies) from the sheer volume of trades. Instead of compiling unique ASTs for every trade, we build a standardized grid of pricing factors (Discount Factors and Annuities) at discrete tenors, and use relational JOINs to apply these factors to the trade portfolio.

### 1. Base Curve Evaluation
We define a finite set of standard benchmark tenors, `DISCRETE_TENORS` (e.g., 1Y, 2Y, 3Y, 5Y, 10Y, 30Y). The `py-flow` reactive engine dynamically generates the AST math columns *only* for the benchmark instruments at these tenors. For each scenario and each discrete tenor, we evaluate two core metrics:
- **Base Discount Factor (DF):** The discount factor for that tenor.
- **Base Annuity:** Extracted analytically from a standard Fixed vs Float swap at that tenor.

### 2. Array Unrolling and Ungrouping
Within the analytical engine (Deephaven or DuckDB), the evaluated scalar columns for the discrete tenors are first gathered into array columns per `Scenario_Id`. We then apply an `ungroup()` (or unpivot) operation to explode the arrays into a narrow, "skinny" table format: `t_scenarios_narrow`.

**Schema of `t_scenarios_narrow`:**
- `Scenario_Id` (int)
- `Tenor_Curve` (double)
- `DF` (double)
- `Annuity` (double)

This table has `NUM_SCENARIOS * NUM_DISCRETE_TENORS` rows, making it highly compact and efficient for analytical engines to serialize, cache, and process.

### 3. The Trade Portfolio ("Swaps" Table)
A separate table represents the analytical risk representation of the actual portfolio, where each swap is pre-mapped to the standard `DISCRETE_TENORS`. 

**Schema of `t_swaps`:**
- `SwapName` (String)
- `Tenor_Swap` (double)
- `FixedRate` (double)
- `Notional` (double)

### 4. Vectorized Pricing JOIN
To value the portfolio across all scenarios simultaneously, the engine performs a standard relational equi-join between the Skinny Table and the Swap Portfolio:

```python
t_eval = t_scenarios_narrow.join(t_swaps, on=["Tenor_Curve = Tenor_Swap"])
```

After the join, a unified vectorized expression computes the Net Present Value (NPV) across all scenario-trade combinations:

```python
NPV = Notional * (FixedRate * Annuity * 10000.0 - (1.0 - DF))
```
*(Note: `1.0 - DF` acts as the NPV of the floating leg, eliminating the need to explicitly price floating cashflows individually).*

## Advantages
1. **O(1) AST Compilation:** The AST generation time is strictly determined by the number of discrete tenors (e.g., 10), entirely independent of the portfolio size.
2. **High-Performance Scaling:** Offloads the heavy lifting of portfolio evaluation to Deephaven/DuckDB, leveraging their highly optimized, vectorized column math and relational JOINS.
3. **Reduced Data Payload:** Minimizes code-generation size and the data movement required between Py-Flow and the execution server.
4. **Memory Efficiency:** "Skinny table" joins exhibit simpler memory access patterns, are easier to partition/shard, and compress exceptionally well compared to extremely wide single-row schemas.

## Limitations of the Simplistic Approach
The initial "discrete tenors" approach (bucketing swaps into a few fixed tenors like 1Y, 5Y, 10Y) will not work fully in a realistic production environment. Real-world portfolios expose several key limitations:

1. **Continuous Tenors & Stubs:** Real swap tenors vary continuously (e.g., 0.5 to 20 years, in increments of ~1/250 to mimic business days). Furthermore, standard scheduling walks backward from the maturity date, creating random short front stubs. This means the required discount factors fall on a massive variety of times, making it impossible to bucket SQL joins purely on the swap's overall tenor.
2. **Exploding AST Complexity:** We cannot generate or compile a unique AST for every distinct cashflow date across a 100,000+ instrument portfolio.
3. **Massive Market Data Universe:** The market data universe is immense. Not only are there many currencies, each with multiple curves (Discount, OIS, Libor) and numerous knots per curve, but asset classes like equities and bonds have an extreme number of market data points. It is fundamentally impossible to represent the entire market data universe as individual database columns.

## The Advanced Skinny Table Architecture

To address these limitations, the architecture must transition to a true "Skinny Table" approach that leverages curve knots and sensitivity weights, fully divorcing instrument definitions from scenario columns.

### 1. Market Data as Rows, Not Columns
Instead of generating columns for each market data point, market quotes and curve knots must be represented as rows. The scenario table should serve the coordinates of market data as primary keys in a skinny column format:
- `Scenario_Id`
- `MarketData_Id` (or `Curve_Knot_Id`)
- `Value` (e.g., rate, shift, or simulated quote)

### 2. Instrument-to-Knot Weight Mapping
Since the database needs some fixed structure to keep the SQL/AST compilation stable, the fixed elements should be the **market data knots** (e.g., `IR_USD_DISC_USD.1Y`, `IR_USD_DISC_USD.2Y`) rather than the swap tenors.

To price the portfolio dynamically without unrolling exact cashflow math in the DB:
1. **Pre-calculate Sensitivities (Weights):** Outside the scenario engine (in `py-flow`), the exact swap definition (including all stub dates, day count fractions, etc.) is processed to compute a table of weights (sensitivities) relative to the underlying curve knots. 
2. **Standardized Risk Representation:** A given 7.3-year swap is reduced from a complex cashflow schedule into a simple vector of weights on the 7Y and 10Y curve knots.
3. **The Pricing Join:** The `t_swaps` table is transformed into `t_swap_sensitivities` (columns: `Swap_Id`, `Curve_Knot_Id`, `Weight`). Pricing a million swaps across a million scenarios becomes a massive grouped aggregation of a skinny join:
   ```sql
   -- Conceptual aggregation
   NPV = SUM(Swap_Knot_Weight * Scenario_Knot_Value) GROUP BY Scenario_Id, Swap_Id
   ```

This advanced architecture ensures that the compiled AST and database schema remain perfectly stable regardless of how many unique exact dates are in the portfolio, scaling seamlessly to 100,000+ instruments and massive market data universes.
