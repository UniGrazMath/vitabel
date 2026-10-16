"""ipywidgets controls of the interactive matplotlib plot, bound to the
interaction core: widget changes become commands, core state is shown in
the widgets."""

from __future__ import annotations

from contextlib import contextmanager

import ipywidgets as widgets

from vitabel.plotting.interaction import InteractionCore, Mode
from vitabel.timeseries import IntervalLabel

ANNOTATE_TAB, SHIFT_TAB, SETTINGS_TAB = 0, 1, 2

ANNOTATE_HELP = """
<p>
Right-click to add data points to the active label. Use the number
keys (<kbd style="color: black;">0</kbd> to <kbd style="color: black;">9</kbd>)
to quickly switch between active labels. Use the button below or the
<kbd style="color: black;">D</kbd> key to toggle between adding and deleting labels.
<kbd style="color: black;">Ctrl</kbd>+<kbd style="color: black;">Z</kbd> undoes the last
change of a label or time shift, <kbd style="color: black;">Ctrl</kbd>+<kbd style="color: black;">Y</kbd>
redoes it.
</p>
<p>
Use the left and right arrow keys to shift the plotted time window.
By clicking in an overview plot (if one is present), the plot window location
is moved.
The <kbd style="color: black;">+</kbd> and <kbd style="color: black;">-</kbd>
keys zoom in and out, respectively.
Mouse scrolling in a subplot zooms the vertical axis.
</p>
"""

SHIFT_HELP = """
<p>Select the channels to be shifted in the menu below (labels attached to
them move along). Multiple channels can be selected by holding down
<kbd style="color: black;">Ctrl</kbd> (on MacOS: <kbd style="color: black;">Cmd</kbd>)
while clicking.
</p>
<p>Right-click into one of the plots to first set a reference time
(represented by a vertical dotted red line), then right-click again
to the position where the reference time should be moved to for the
selected channels. Press <kbd style="color: black;">Esc</kbd> to
clear a set reference time.</p>
"""


def _aligned_label(text: str) -> widgets.Label:
    return widgets.Label(
        text,
        layout=widgets.Layout(
            min_width="85px",
            margin="0px 5px 0px 0px",
            justify_content="flex-end",
        ),
    )


class ControlPanel:
    """The tabs (annotate, align timelines, settings) above the figure.

    Parameters
    ----------
    core
        The interaction core.
    """

    def __init__(self, core: InteractionCore):
        self.core = core
        self._updating = False

        # -- annotation ------------------------------------------------------------
        self.label_dropdown = widgets.Dropdown(
            options=[
                (f"{label.name}   [Interval]", index)
                if isinstance(label, IntervalLabel)
                else (label.name, index)
                for index, label in enumerate(core.labels, start=1)
            ],
            disabled=False,
            layout=widgets.Layout(margin="3px 0px 0px 0px"),
        )
        self.delete_toggle = widgets.ToggleButton(
            value=False,
            description="Mode: Add Data",
            disabled=False,
            button_style="success",
        )
        checkbox = dict(
            description="", indent=False, layout=widgets.Layout(width="max-content")
        )
        self.timestamp_check = widgets.Checkbox(value=True, disabled=True, **checkbox)
        self.value_check = widgets.Checkbox(value=True, disabled=False, **checkbox)
        self.text_check = widgets.Checkbox(value=True, disabled=False, **checkbox)
        self.text_input = widgets.Text(placeholder="Label text ...")
        self.text_stack = widgets.Stack([widgets.HTML(), self.text_input])
        self.text_stack.selected_index = 1
        self.add_stack = widgets.Stack(
            [
                widgets.HTML(),
                widgets.HBox(
                    [
                        _aligned_label("Data to add:"),
                        widgets.HBox(
                            [
                                widgets.VBox(
                                    [widgets.Label("Timestamp"), self.timestamp_check],
                                    layout=widgets.Layout(margin="0 20px 0 0"),
                                ),
                                widgets.VBox(
                                    [widgets.Label("Numeric value"), self.value_check],
                                    layout=widgets.Layout(margin="0 20px 0 0"),
                                ),
                                widgets.VBox(
                                    [
                                        widgets.Label("Textual value"),
                                        widgets.HBox(
                                            [self.text_check, self.text_stack]
                                        ),
                                    ]
                                ),
                            ],
                            layout=widgets.Layout(
                                border="1px dotted gray", padding="3px 8px"
                            ),
                        ),
                    ]
                ),
            ],
            layout=widgets.Layout(margin="3px 0 0 0 "),
        )
        self.add_stack.selected_index = 1

        # -- shifting --------------------------------------------------------------
        channels = core.spec.collection.channels
        self.shift_selection = widgets.SelectMultiple(
            value=channels[:1],
            options=[
                (f"[{index}] {channel.name}", channel)
                for index, channel in enumerate(channels)
            ],
            description="Channels",
            disabled=False,
        )

        # -- settings --------------------------------------------------------------
        self.limit_widgets = [
            widgets.HBox(
                [
                    widgets.Label(f"Plot {index} limits:"),
                    widgets.FloatText(value=0, description="min"),
                    widgets.FloatText(value=1, description="max"),
                ]
            )
            for index, _ in enumerate(core.spec.channels, start=1)
        ]
        self.apply_button = widgets.Button(description="Apply")

        self.tab = widgets.Tab()
        self.tab.children = [
            widgets.VBox(
                [
                    widgets.HTML(ANNOTATE_HELP),
                    widgets.HBox(
                        [
                            _aligned_label("Label:"),
                            self.label_dropdown,
                            self.delete_toggle,
                        ]
                    ),
                    self.add_stack,
                ]
            ),
            widgets.VBox([widgets.HTML(SHIFT_HELP), self.shift_selection]),
            widgets.VBox(
                [
                    widgets.Label("Vertical plot limits:"),
                    *self.limit_widgets,
                    self.apply_button,
                ]
            ),
        ]
        self.tab.titles = ["Annotate", "Align Timelines", "Settings"]
        self.tab.layout = widgets.Layout(min_height="270px")

        self._show_options()
        self._bind()

    # -- binding -------------------------------------------------------------------

    @contextmanager
    def _updating_widgets(self):
        """Widget changes made to reflect the core state are not commands."""
        self._updating, previous = True, self._updating
        try:
            yield
        finally:
            self._updating = previous

    def _command(self, function):
        def handler(change):
            if not self._updating:
                function(change)

        return handler

    def _bind(self) -> None:
        core = self.core
        self.label_dropdown.observe(
            self._command(lambda change: core.select_label(change["new"] - 1)),
            names="value",
        )
        self.delete_toggle.observe(
            self._command(lambda change: core.set_mode(self.mode())), names="value"
        )
        self.tab.observe(
            self._command(lambda change: core.set_mode(self.mode())),
            names="selected_index",
        )
        self.value_check.observe(
            self._command(lambda change: core.set_options(use_value=change["new"])),
            names="value",
        )
        self.text_check.observe(
            self._command(lambda change: core.set_options(use_text=change["new"])),
            names="value",
        )
        self.text_input.observe(
            self._command(lambda change: core.set_options(text=change["new"])),
            names="value",
        )
        self.shift_selection.observe(
            self._command(lambda change: core.set_shift_channels(change["new"])),
            names="value",
        )
        self.apply_button.on_click(lambda button: self._apply_limits())
        self.tab.observe(
            lambda change: change["new"] == SETTINGS_TAB and self.on_settings_opened(),
            names="selected_index",
        )
        core.on("state_changed", self._on_state_changed)

    def mode(self) -> Mode:
        """The mode selected by tab and delete toggle."""
        if self.tab.selected_index == SHIFT_TAB:
            return "shift"
        if self.tab.selected_index == SETTINGS_TAB:
            return "none"
        return "delete" if self.delete_toggle.value else "annotate"

    def toggle_delete(self) -> None:
        """Toggle between adding and deleting (``d`` key)."""
        self.delete_toggle.value = not self.delete_toggle.value

    def _apply_limits(self) -> None:
        for subplot, limit_widget in enumerate(self.limit_widgets):
            _, low, high = limit_widget.children
            self.core.set_ylim(subplot, low.value, high.value)

    def on_settings_opened(self) -> None:
        """Called when the settings tab is opened (set by the plot)."""

    def show_limits(self, limits: list[tuple[float, float]]) -> None:
        for limit_widget, (low, high) in zip(self.limit_widgets, limits):
            _, low_input, high_input = limit_widget.children
            low_input.value, high_input.value = low, high

    # -- core state → widgets ------------------------------------------------------

    def _on_state_changed(self, field: str) -> None:
        with self._updating_widgets():
            if field == "options":
                self._show_options()
            elif field == "active_label":
                label = self.core.active_label
                if label is not None:
                    self.label_dropdown.value = self.core.labels.index(label) + 1
            elif field == "mode":
                self._show_mode()

    def _show_options(self) -> None:
        options = self.core.options
        with self._updating_widgets():
            self.value_check.value = options.use_value
            self.text_check.value = options.use_text
            self.text_input.value = options.text
            self.text_stack.selected_index = 1 if options.use_text else 0

    def _show_mode(self) -> None:
        mode = self.core.mode
        if mode in ("annotate", "delete"):
            self.tab.selected_index = ANNOTATE_TAB
            self.delete_toggle.value = mode == "delete"
        else:  # the delete toggle keeps its state for the annotate tab
            self.tab.selected_index = SHIFT_TAB if mode == "shift" else SETTINGS_TAB
        deleting = self.delete_toggle.value
        self.delete_toggle.description = (
            "Mode: Delete Data" if deleting else "Mode: Add Data"
        )
        self.delete_toggle.button_style = "danger" if deleting else "success"
        self.add_stack.selected_index = 0 if deleting else 1
