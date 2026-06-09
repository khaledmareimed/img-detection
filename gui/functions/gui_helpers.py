"""
GUI theme constants and helper utilities for CustomTkinter.
"""

import customtkinter as ctk

# --- Global CTk appearance ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Colour palette ---
COLORS = {
    "bg":          "#0f172a",
    "bg_card":     "#1e293b",
    "bg_input":    "#334155",
    "accent":      "#6366f1",
    "accent_hover":"#818cf8",
    "success":     "#22c55e",
    "error":       "#ef4444",
    "warning":     "#f59e0b",
    "cyan":        "#06b6d4",
    "text":        "#f1f5f9",
    "text_dim":    "#94a3b8",
    "border":      "#334155",
}

# --- Font presets ---
FONT_TITLE   = ("Segoe UI", 14, "bold")
FONT_HEADING = ("Segoe UI", 12, "bold")
FONT_BODY    = ("Segoe UI", 11)
FONT_SMALL   = ("Segoe UI", 10)
FONT_MONO    = ("Cascadia Code", 10)


def styled_button(
    parent: ctk.CTkFrame,
    text: str,
    command,
    color: str = COLORS["accent"],
    hover: str = COLORS["accent_hover"],
    **kwargs,
) -> ctk.CTkButton:
    """Create a consistently styled button."""
    return ctk.CTkButton(
        parent, text=text, command=command,
        fg_color=color, hover_color=hover,
        font=FONT_BODY, corner_radius=8, height=36,
        **kwargs,
    )
