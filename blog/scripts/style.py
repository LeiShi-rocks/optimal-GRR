"""Shared colors and matplotlib style for all blog figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2b6cb0"     # design variance / GER-calibrated estimators
ORANGE = "#dd6b20"   # GER estimates / our method's highlights
GRAY = "#8a8f98"     # prediction error / uncalibrated baselines
DARK = "#2d3748"     # Hajek reference
RED = "#c53030"      # failure cases


def setup():
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "svg.fonttype": "none",
        "legend.fontsize": 9,
    })


def save(fig, path_svg):
    """Write the SVG used by the article and a PNG preview next to it."""
    fig.savefig(path_svg)
    fig.savefig(path_svg[:-4] + ".png", dpi=150)
