import panel as pn
import param
import bokeh.plotting as bp
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Union

from dashboard.theme import apply as apply_theme

# ── Filter State ────────────────────────────────────────────────────────────

class FilterState(param.Parameterized):
    """Global (per-page) reactive storage for widget-driven filters.
    
    A Tabulator or Chart can bind to a key in this state to auto-refresh 
    when the user changes a widget selection.
    """
    values = param.Dict(default={}, doc="Current filter key-value pairs.")

    def set(self, key: str, value: Any):
        """Update a filter and trigger watchers."""
        new_values = dict(self.values)
        new_values[key] = value
        self.values = new_values

# ── Components ──────────────────────────────────────────────────────────────

class _Component:
    """Base class for dashboard UI elements (Tables, Charts, HTML)."""
    def __init__(self, obj: Any, title: str = "", span: int = 6, height: int = 300):
        self.obj = obj
        self.title = title
        self.span = span
        self.height = height
        
        # We wrap in a 'card' style (CSS class from theme.py)
        # Using pn.Column to provide a container for title + content
        header = pn.pane.HTML(f'<div class="kpi-title">{title}</div>', margin=(0, 0, 8, 0)) if title else ""
        self.layout = pn.Column(
            header,
            obj,
            css_classes=["card"],
            sizing_mode="stretch_width",
            height=height,
            margin=8
        )

class _TableComponent(_Component):
    """Wraps a pn.widgets.Tabulator with .update() logic."""
    def update(self, df: pd.DataFrame):
        self.obj.value = df

class _ChartComponent(_Component):
    """Wraps a Bokeh figure pane."""
    def update(self, fig: Any):
        self.obj.object = fig

# ── Page ──────────────────────────────────────────────────────────────────

class Page(param.Parameterized):
    """A single view within the Dashboard, managing its own components and layout."""
    
    def __init__(self, name: str, icon: str = "", **params):
        super().__init__(name=name, **params)
        self.icon = icon
        self.components: List[_Component] = []
        self.filters = FilterState()
        self._layout = pn.FlexBox(justify_content="start", align_items="start", sizing_mode="stretch_width")

    def add_table(self, df: pd.DataFrame, title: str = "", span: int = 12, height: int = 400, **kwargs) -> _TableComponent:
        """Add a searchable, sortable data table."""
        theme = kwargs.pop("theme", "fast") # Tabulator theme
        tbl = pn.widgets.Tabulator(df, theme=theme, sizing_mode="stretch_both", **kwargs)
        comp = _TableComponent(tbl, title=title, span=span, height=height)
        self.components.append(comp)
        return comp

    def add_chart(self, fig: Any, title: str = "", span: int = 6, height: int = 400) -> _ChartComponent:
        """Add a Bokeh or Plotly chart."""
        pane = pn.pane.Bokeh(fig, sizing_mode="stretch_both") if hasattr(fig, 'renderers') else pn.pane.Plotly(fig)
        comp = _ChartComponent(pane, title=title, span=span, height=height)
        self.components.append(comp)
        return comp

    def add_html(self, html: str, title: str = "", span: int = 4, height: int = 200) -> _Component:
        """Add raw HTML (useful for KPI tiles)."""
        pane = pn.pane.HTML(html, sizing_mode="stretch_both")
        comp = _Component(pane, title=title, span=span, height=height)
        self.components.append(comp)
        return comp

    def add_widget(self, widget: pn.widgets.Widget, title: str = "", span: int = 3) -> _Component:
        """Add a control widget (Select, Slider, etc)."""
        comp = _Component(widget, title=title, span=span, height=100)
        self.components.append(comp)
        return comp

    def bind_filter(self, widget: pn.widgets.Widget, key: str):
        """Bind a widget's value to the page's FilterState."""
        def _update_filter(event):
            self.filters.set(key, event.new)
        widget.param.watch(_update_filter, "value")
        # Initialize
        self.filters.set(key, widget.value)

    @property
    def layout(self) -> pn.FlexBox:
        """Assemble the 12-column grid layout."""
        # We map span 1..12 to percentage widths for the FlexBox
        self._layout.objects = []
        for comp in self.components:
            # We wrap the card to define its flex width
            width_pct = (comp.span / 12) * 100
            wrapper = pn.Column(comp.layout, width=int(width_pct), min_width=250, sizing_mode="stretch_width")
            # Force basis to percentage for responsive grid
            wrapper.styles = {"flex-basis": f"calc({width_pct}% - 16px)"}
            self._layout.append(wrapper)
        return self._layout

# ── Dashboard ───────────────────────────────────────────────────────────────

class Dashboard(param.Parameterized):
    """Top-level Dashboard container with sidebar navigation and multi-page support."""
    
    active_page_name = param.String(default="")
    
    def __init__(self, title: str, subtitle: str = "", live: bool = True, **params):
        super().__init__(name=title, **params)
        self.subtitle = subtitle
        self.live = live
        self.pages: Dict[str, Page] = {}
        
        apply_theme()
        
        # UI Structure
        self._sidebar = pn.Column(
            pn.pane.HTML(f'<h2>{title}</h2><p style="color:#9ca3af">{subtitle}</p>', margin=(20, 20)),
            sizing_mode="stretch_height"
        )
        self._main_content = pn.Column(sizing_mode="stretch_both", margin=20)
        
    def add_page(self, name: str, icon: str = "") -> Page:
        """Create and register a new page."""
        page = Page(name, icon=icon)
        self.pages[name] = page
        
        if not self.active_page_name:
            self.active_page_name = name
            
        # Create a navigation button
        btn = pn.widgets.Button(name=f"{icon}  {name}" if icon else name, 
                                 button_type="default", 
                                 sizing_mode="stretch_width",
                                 margin=(4, 20))
        
        def _on_click(event):
            self.active_page_name = name
            
        btn.on_click(_on_click)
        self._sidebar.append(btn)
        
        return page

    @pn.depends("active_page_name")
    def _render_page(self):
        if not self.active_page_name or self.active_page_name not in self.pages:
            return pn.pane.HTML("No page selected")
        return self.pages[self.active_page_name].layout

    def serve(self, port: int = 8050, threaded: bool = False):
        """Serve the dashboard as a standalone web app."""
        template = pn.template.BootstrapTemplate(
            title=self.name,
            sidebar=[self._sidebar],
            main=[self._render_page]
        )
        return pn.serve(template, port=port, threaded=threaded, show=not threaded)

    def show(self):
        """Return the servable template (for Jupyter usage)."""
        template = pn.template.BootstrapTemplate(
            title=self.name,
            sidebar=[self._sidebar],
            main=[self._render_page]
        )
        return template.servable()
