import logging
import os

import seaborn as sns

logger = logging.getLogger(__name__)

# Shared ink/grid colors for paper figures: near-black text, muted axes, and a
# warm light-gray grid that recedes behind the data.
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e1e0d9'

# Unified strategy palette for ALL exploration figures (single source of truth).
# Hiroshige-family tones (MetBrewer): warm coral / soft amber / deep steel blue —
# the muted-yet-rich warm-cool language of contemporary Nature figures;
# distinguishable under deuteranopia and ordered in grayscale luminance.
STRATEGY_PALETTE = {
    'explorer': '#E76254',           # warm coral
    'exploiter': '#F2A65A',          # soft amber
    'cautious_explorer': '#376795',  # deep steel blue
}
STRATEGY_NICE = {'explorer': 'Explorer', 'exploiter': 'Exploiter',
                 'cautious_explorer': 'Cautious explorer'}

# Accent pair for two-series figures (e.g., dual-axis panels)
ACCENT_WARM, ACCENT_COOL = '#E76254', '#1E466E'

# Vivid accent trio for the resubmission and review-bias figures (Figs 7/8):
# cobalt / NPG red / gold, brighter than the Hiroshige pair, still CVD-safe.
VIVID_COOL, VIVID_WARM, VIVID_GOLD = '#1F6FB5', '#E5493A', '#E8A33D'


def nature_style():
    """Nature-journal seaborn theme shared by all paper figures.

    Sans-serif 6-7.5 pt type, thin 0.6 pt axes, bold left-aligned panel
    titles, and editable text in exported PDF/SVG (fonttype 42).
    """
    sns.set_theme(style='ticks', context='paper', rc={
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6.5,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'axes.titlelocation': 'left', 'axes.titleweight': 'bold',
        'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.labelcolor': INK, 'text.color': INK,
        'axes.unicode_minus': False,  # Arial lacks U+2212; use ASCII hyphen
        'pdf.fonttype': 42, 'svg.fonttype': 'none',  # keep text editable
    })
