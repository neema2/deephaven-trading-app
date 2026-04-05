import panel as pn

# ── Financial Dark Palette ──────────────────────────────────────────────────
PALETTE = {
    "bg_primary": "#0f1117",      # deep navy/black
    "bg_secondary": "#1a1c23",    # card background
    "bg_sidebar": "#16181d",      # sidebar
    "accent": "#3b82f6",          # blue
    "green": "#10b981",           # emerald
    "red": "#ef4444",             # rose
    "yellow": "#f59e0b",          # amber
    "gray": "#6b7280",            # muted text
    "text": "#f3f4f6",            # primary text
    "border": "#2d303d",          # component borders
}

# ── Bokeh Theme Overrides ────────────────────────────────────────────────────
BOKEH_THEME = {
    "attrs": {
        "Figure": {
            "background_fill_color": PALETTE["bg_secondary"],
            "border_fill_color": PALETTE["bg_primary"],
            "outline_line_color": PALETTE["border"],
            "title_text_color": PALETTE["text"],
            "min_border": 20,
        },
        "Axis": {
            "axis_line_color": PALETTE["gray"],
            "major_label_text_color": PALETTE["gray"],
            "major_tick_line_color": PALETTE["gray"],
            "minor_tick_line_color": None,
            "axis_label_text_color": PALETTE["text"],
            "axis_label_text_font_size": "10pt",
        },
        "Grid": {
            "grid_line_color": "#242731",
            "grid_line_alpha": 1.0,
        },
        "Legend": {
            "background_fill_color": PALETTE["bg_sidebar"],
            "label_text_color": PALETTE["text"],
            "border_line_color": PALETTE["border"],
        }
    }
}

# ── Global CSS Overrides ─────────────────────────────────────────────────────
GLOBAL_CSS = """
/* Reset & Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

body {
    background-color: #0f1117 !important;
    color: #f3f4f6 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Card Style */
.bk .card {
    background-color: #1a1c23 !important;
    border: 1px solid #2d303d !important;
    border-radius: 8px !important;
    padding: 16px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
}

/* Sidebar */
.pn-sidebar {
    background-color: #16181d !important;
    border-right: 1px solid #2d303d !important;
}

/* Tabulator Overrides */
.tabulator {
    background-color: transparent !important;
    color: #f3f4f6 !important;
    border: none !important;
}
.tabulator-header {
    background-color: #16181d !important;
    color: #9ca3af !important;
    border-bottom: 2px solid #2d303d !important;
}
.tabulator-row {
    background-color: transparent !important;
    border-bottom: 1px solid #242731 !important;
}
.tabulator-row:hover {
    background-color: #2d303d !important;
}

/* KPIs */
.kpi-card {
    text-align: center;
}
.kpi-title {
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
}
.kpi-value {
    color: #3b82f6;
    font-size: 1.5rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
}
"""

def apply():
    """Apply the financial dark theme globally to the Panel session."""
    import bokeh.themes as bt
    
    # Set Panel theme and extension
    pn.extension(
        "tabulator", "plotly", "mathjax",
        raw_css=[GLOBAL_CSS],
        sizing_mode="stretch_width"
    )
    
    pn.config.template = "bootstrap"
    pn.config.theme = "dark"
    
    # We apply the Bokeh theme custom dictionary
    # (Note: Bokeh 3.x uses different theme application than 2.x)
    # This is a placeholder for session-level application
    pass
