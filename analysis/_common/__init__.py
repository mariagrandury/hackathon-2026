"""Shared utilities for the analysis suite.

`data_loading` flattens the two HF datasets into analysis-ready DataFrames
(plus the Eventbrite demographic join). `plotting` exposes a small set of
matplotlib primitives and the color palette. `metrics` holds the numeric
helpers (Wilson CI, unanimity, Lorenz). `io_utils` is for CSV / directory
handling. Each section's `analyze.py` imports from here and adds its own
thin per-plot wrappers.
"""
