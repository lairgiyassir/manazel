"""
ARCV / W_topo scatter, manual k-NN on hilal_dataset_final.xlsx, and matplotlib figures
for the Streamlit "Details" section.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Headless-safe matplotlib cache dir before importing matplotlib (mirrors scripts/plot_eid_next_day_morocco.py)
_ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = _ROOT / ".tmp_mpl"
    _mpl_cfg.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import utils.astronomy_ as astronomy  # noqa: E402
from utils.odeh import calculate  # noqa: E402

DATASET_PATH = _ROOT / "datasets" / "hilal_dataset_final.xlsx"

GregorianDay = Union[date, datetime, Tuple[int, int, int]]


def _as_date(d: GregorianDay) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    y, m, day = d
    return date(y, m, day)


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Load hilal_dataset_final.xlsx (arcv, W_topo, output)."""
    p = path or DATASET_PATH
    df = pd.read_excel(p)
    required = {"arcv", "W_topo", "output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    return df


def apply_scientific_style() -> None:
    """Publication-friendly matplotlib defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def compute_arcv_wtopo(
    year: int,
    month: int,
    day: int,
    *,
    latitude: float = 34.0084,
    longitude: float = 6.8539,
) -> Optional[Tuple[float, float]]:
    """Return (ARCV, W_topo) for local midnight Gregorian date, or None if undefined."""
    base_time = astronomy.Time.Make(year, month, day, 0, 0, 0)
    result = calculate(base_time=base_time, latitude=latitude, longitude=longitude)
    if "ARCV" not in result or "W_topo" not in result:
        return None
    return float(result["ARCV"]), float(result["W_topo"])


def knn_predict(
    point: Tuple[float, float],
    df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Manual k-NN with z-scored Euclidean distance on (arcv, W_topo).
    Majority vote on `output` (0/1). k must be odd to avoid ties.
    """
    if k % 2 == 0:
        raise ValueError("k must be odd for majority vote without ties")
    cols = ["arcv", "W_topo"]
    mu = df[cols].mean()
    sd = df[cols].std().replace(0.0, 1.0)
    norm = (df[cols] - mu) / sd
    p = (pd.Series({"arcv": point[0], "W_topo": point[1]}) - mu) / sd
    d = np.sqrt(((norm - p) ** 2).sum(axis=1))
    nn = df.assign(_d=d).nsmallest(k, "_d").copy()
    votes_1 = int((nn["output"] == 1).sum())
    votes_0 = k - votes_1
    label = 1 if votes_1 > votes_0 else 0
    return {
        "label": label,
        "votes_1": votes_1,
        "votes_0": votes_0,
        "neighbours": nn,
    }


def format_knn_vote_paragraph(
    knn_d1: Optional[Dict[str, Any]],
    knn_d2: Optional[Dict[str, Any]],
    *,
    d1: date,
    d2: date,
    k: int = 5,
) -> str:
    """
    Beginner-friendly summary of k-NN votes at D-1 and D-2 (for Streamlit UI).
    """

    def _simple_outcome(v1: int, v0: int, lbl: int) -> str:
        """Plain-language outcome for the k-neighbour vote."""
        if lbl == 1 and v1 == k and v0 == 0:
            return (
                f"all {k} of those past nights match the situation where **Morocco went with the next day** "
                f"as the first day of the new month (everyone agrees)."
            )
        if lbl == 0 and v0 == k and v1 == 0:
            return (
                f"all {k} of those past nights match the situation where **Morocco did not** use the next day "
                f"as the first day of the new month in that pattern (everyone agrees)."
            )
        if lbl == 1:
            return (
                f"**most** of the {k} nights ({v1} out of {k}) match the “next day is month start” pattern; "
                f"{v0} do not."
            )
        return (
            f"**most** of the {k} nights ({v0} out of {k}) do **not** match that “next day is month start” pattern; "
            f"{v1} do."
        )

    parts: list[str] = []
    if knn_d1 is not None:
        v1, v0 = knn_d1["votes_1"], knn_d1["votes_0"]
        lbl = int(knn_d1["label"])
        parts.append(
            f"**D-1** ({d1.isoformat()}) is the night *right before* the day we predict as the first day of the new Hijri month—"
            f"the usual “doubt night” when observers check the crescent. "
            f"We compare that night’s moon measurements (two numbers called **ARCV** and **W_topo**, which describe how easy the thin crescent is to see) "
            f"to our past data and find the **{k} past nights** whose measurements are closest (after putting both numbers on the same scale). "
            f"Then we look at what Morocco did on the calendar after each of those nights: {_simple_outcome(v1, v0, lbl)}"
        )
    if knn_d2 is not None:
        v1, v0 = knn_d2["votes_1"], knn_d2["votes_0"]
        lbl = int(knn_d2["label"])
        parts.append(
            f"**D-2** ({d2.isoformat()}) is **two nights before** that same predicted first day. "
            f"We repeat the same “find the {k} closest past nights, then see what happened next” idea. "
            f"{_simple_outcome(v1, v0, lbl)} "
            f"So you can see how the picture **changes** when you go one night earlier than D-1."
        )
    if not parts:
        return ""
    return " ".join(parts)


def build_details_figure(
    predicted_first_day: GregorianDay,
    df: Optional[pd.DataFrame] = None,
    *,
    hijri_year: Optional[int] = None,
    hijri_month_name: Optional[str] = None,
    latitude: float = 34.0084,
    longitude: float = 6.8539,
    k: int = 5,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Scatter of historical (ARCV, W_topo) with output 0/1, overlay doubt-nights D-1 and D-2,
    lines to k nearest neighbours per overlay, and KNN summaries.

    predicted_first_day: Gregorian first day of Hijri month D (from logistic pipeline).
    """
    df = df if df is not None else load_dataset()
    D = _as_date(predicted_first_day)
    d1 = D - timedelta(days=1)
    d2 = D - timedelta(days=2)

    apply_scientific_style()

    positive = df.loc[df["output"] == 1].copy()
    negative = df.loc[df["output"] == 0].copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.9), constrained_layout=False)

    if not negative.empty:
        ax.scatter(
            negative["arcv"],
            negative["W_topo"],
            s=40,
            c="#c41e3a",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.8,
            zorder=2,
            label=rf"Other ($n={len(negative)}$, output = 0)",
        )
    if not positive.empty:
        ax.scatter(
            positive["arcv"],
            positive["W_topo"],
            s=42,
            c="#0b3d91",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.85,
            zorder=2,
            label=rf"Next-day start ($n={len(positive)}$, output = 1)",
        )

    details: Dict[str, Any] = {
        "predicted_first_day": (D.year, D.month, D.day),
        "doubt_night_d1": (d1.year, d1.month, d1.day),
        "doubt_night_d2": (d2.year, d2.month, d2.day),
        "d1": d1,
        "d2": d2,
        "knn_d1": None,
        "knn_d2": None,
        "point_d1": None,
        "point_d2": None,
    }

    # D-1 overlay (gold star) + KNN lines
    pt1 = compute_arcv_wtopo(d1.year, d1.month, d1.day, latitude=latitude, longitude=longitude)
    if pt1 is not None:
        arcv1, w1 = pt1
        details["point_d1"] = {"arcv": arcv1, "W_topo": w1}
        knn1 = knn_predict((arcv1, w1), df, k=k)
        details["knn_d1"] = {
            **knn1,
            "neighbours": knn1["neighbours"],  # keep DataFrame for caller if needed
        }
        lbl1 = f"{d1.isoformat()} (D-1, doubt night)"
        ax.scatter(
            [arcv1],
            [w1],
            s=240,
            marker="*",
            c="#f5b301",
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label=lbl1,
        )
        ax.annotate(
            d1.isoformat(),
            (arcv1, w1),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.92),
            zorder=7,
        )
        for _, row in knn1["neighbours"].iterrows():
            ax.plot(
                [arcv1, row["arcv"]],
                [w1, row["W_topo"]],
                color="#2ca02c",
                alpha=0.45,
                linewidth=1.0,
                zorder=3,
            )
        # Hollow markers (point size in **points**), so rings stay visually round unlike data-space Circle patches.
        nbr1 = knn1["neighbours"]
        ax.scatter(
            nbr1["arcv"],
            nbr1["W_topo"],
            s=160,
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=1.35,
            alpha=0.9,
            zorder=4,
        )

    # D-2 overlay (orange diamond) + KNN lines
    pt2 = compute_arcv_wtopo(d2.year, d2.month, d2.day, latitude=latitude, longitude=longitude)
    if pt2 is not None:
        arcv2, w2 = pt2
        details["point_d2"] = {"arcv": arcv2, "W_topo": w2}
        knn2 = knn_predict((arcv2, w2), df, k=k)
        details["knn_d2"] = {**knn2, "neighbours": knn2["neighbours"]}
        lbl2 = f"{d2.isoformat()} (D-2)"
        ax.scatter(
            [arcv2],
            [w2],
            s=130,
            marker="D",
            c="#ff7f0e",
            edgecolors="black",
            linewidths=0.9,
            zorder=6,
            label=lbl2,
        )
        ax.annotate(
            d2.isoformat(),
            (arcv2, w2),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.5", alpha=0.92),
            zorder=7,
        )
        for _, row in knn2["neighbours"].iterrows():
            ax.plot(
                [arcv2, row["arcv"]],
                [w2, row["W_topo"]],
                color="#9467bd",
                alpha=0.45,
                linewidth=1.0,
                linestyle=":",
                zorder=3,
            )
        nbr2 = knn2["neighbours"]
        ax.scatter(
            nbr2["arcv"],
            nbr2["W_topo"],
            s=150,
            facecolors="none",
            edgecolors="#9467bd",
            linewidths=1.35,
            alpha=0.9,
            zorder=4,
        )

    ax.set_xlabel(r"ARCV ($^\circ$)")
    ax.set_ylabel(r"$W_{\mathrm{topo}}$ (arcmin)")
    title = "Morocco: crescent parameters vs. month-start label"
    if hijri_year is not None and hijri_month_name:
        title += f"\nHijri context: {hijri_month_name} {hijri_year} — predicted 1st day (Gregorian): {D.isoformat()}"
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.6", loc="best")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.90, bottom=0.12)

    return fig, details
