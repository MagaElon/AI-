# AI-
#Foundations of AI
from __future__ import annotations

import io
import sys
import math
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:  # pragma: no cover
    HAS_CARTOPY = False

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.grid"] = True

USGS_BASE = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
)
DEFAULT_PERIOD = "week"
DEFAULT_MINMAG = 2.5

OUT_CSV = "earthquakes.csv"
OUT_TOP10 = "earthquakes_top10.csv"
OUT_HIST = "hist_magnitude.png"
OUT_MAP = "map_quakes.png"



def usgs_url(period: str, minmag: float | str) -> str:
    if isinstance(minmag, (int, float)):
        mag_tag = f"{float(minmag):.1f}"
    else:
        mag_tag = str(minmag)
    return f"{USGS_BASE}/{mag_tag}_{period}.csv"


def read_usgs(period: str = DEFAULT_PERIOD,
              minmag: float | str = DEFAULT_MINMAG) -> pd.DataFrame:
    url = usgs_url(period, minmag)
    df = pd.read_csv(url)
    cols = [
        "time", "latitude", "longitude", "depth", "mag", "place",
        "type", "nst", "gap", "rms", "id"
    ]
    df = df.loc[:, [c for c in cols if c in df.columns]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["mag", "latitude", "longitude"])
    return df


def magnitude_bins(df: pd.DataFrame,
                   bin_width: float = 0.5) -> pd.DataFrame:
    mag = df["mag"].to_numpy()
    min_m = math.floor(mag.min() / bin_width) * bin_width
    max_m = math.ceil(mag.max() / bin_width) * bin_width
    bins = np.arange(min_m, max_m + bin_width, bin_width)
    cats = pd.cut(mag, bins=bins, right=False)
    table = cats.value_counts().sort_index().rename_axis("bin")
    return table.reset_index(name="count")


def top_k_by_magnitude(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    cols = ["time", "mag", "latitude", "longitude", "depth", "place", "id"]
    cols = [c for c in cols if c in df.columns]
    out = df.sort_values("mag", ascending=False).head(k).loc[:, cols].copy()
    return out.reset_index(drop=True)
def plot_histogram(df: pd.DataFrame,
                   out_path: str = OUT_HIST) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(df["mag"],
            bins=np.arange(df["mag"].min(),
                           df["mag"].max() + 0.2, 0.2),
            edgecolor="black", alpha=0.8)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Count")
    ax.set_title("Earthquake Magnitude Histogram")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_world_map(df: pd.DataFrame,
                   out_path: str = OUT_MAP) -> None:
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    mags = df["mag"].to_numpy()

    size = np.clip((mags - mags.min() + 0.1) * 10.0, 4, 40)

    if HAS_CARTOPY:
        proj = ccrs.Robinson()
        fig = plt.figure(figsize=(7.6, 3.8))
        ax = plt.axes(projection=proj)
        ax.add_feature(cfeature.LAND, facecolor="#f2efe6")
        ax.add_feature(cfeature.OCEAN, facecolor="#cad8f6")
        ax.coastlines(linewidth=0.6)
        ax.gridlines(draw_labels=False, linewidth=0.25, alpha=0.6)
        ax.scatter(
            lons, lats, s=size, c=mags, cmap="plasma", alpha=0.75,
            transform=ccrs.PlateCarree(), edgecolors="k", linewidths=0.2
        )
        ax.set_title("Earthquakes — past week (USGS)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.scatter(lons, lats, s=size, c=mags, cmap="plasma",
               alpha=0.75, edgecolors="k", linewidths=0.2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Earthquakes — past week (USGS) [no cartopy]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
def main(period: str = DEFAULT_PERIOD,
         minmag: float | str = DEFAULT_MINMAG) -> None:
    print("Fetching USGS data ...")
    df = read_usgs(period=period, minmag=minmag)
    print(f"Rows: {len(df)}")

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote: {OUT_CSV}")

    utc_now = datetime.now(timezone.utc)
    print("Now (UTC):", utc_now.strftime("%Y-%m-%d %H:%M"))
    print("Time span:",
          df["time"].min().strftime("%Y-%m-%d %H:%M"),
          "→",
          df["time"].max().strftime("%Y-%m-%d %H:%M"))
    print("Mag range:",
          f"{df['mag'].min():.2f} .. {df['mag'].max():.2f}")

    top10 = top_k_by_magnitude(df, k=10)
    top10.to_csv(OUT_TOP10, index=False)
    print(f"Wrote: {OUT_TOP10}")

    plot_histogram(df, OUT_HIST)
    print(f"Wrote: {OUT_HIST}")
    plot_world_map(df, OUT_MAP)
    print(f"Wrote: {OUT_MAP}")

    display(top10.head())


main(period="week", minmag=2.5)
