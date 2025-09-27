"""
MECE Segmentation for Cart Abandonment Retention Strategy

- Reads an input CSV with fields:
  user_id, cart_abandoned_date, last_order_date, avg_order_value, sessions_last_30d,
  num_cart_items, engagement_score, profitability_score

- Filters universe to users who abandoned a cart in the last N days (default 7)
  relative to execution date.

- Computes quantile thresholds for key features and applies a mutually exclusive,
  collectively exhaustive (MECE) rule set to assign exactly one segment per user.

- Enforces min and max segment sizes. Small segments are merged into "Other".
  Oversized segments are split into parts (deterministic partitioning by user_id).

- Computes segment metrics (conversion_potential, profitability, size, strategic_fit)
  and an overall weighted score. Outputs a CSV or JSON summary.

Usage:
  python mece_cart_segmentation.py --input data.csv --output segments.csv
  python mece_cart_segmentation.py --input data.csv --output segments.json

Configurable constants appear near the top of this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from math import ceil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import Timestamp


# =========================
# Configurable constants
# =========================

# Universe filter
LOOKBACK_DAYS = 7

# Segment size constraints
MIN_SEGMENT_SIZE = 500
MAX_SEGMENT_SIZE = 20000
SPLIT_OVERSIZED_SEGMENTS = True  # if True, splits oversized segments into parts

# Threshold quantiles
QUANTILE_LOW = 0.20
QUANTILE_HIGH = 0.80
SESSIONS_HIGH_Q = 0.75
ITEMS_HIGH_Q = 0.80

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "conversion_potential": 0.30,
    "profitability": 0.25,
    "size": 0.20,
    "strategic_fit": 0.25,
}

# Logging
LOG_LEVEL = logging.INFO


def _setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MECE segmentation for cart abandonment")
    # Make input/output optional; default to sample_cart_abandoners.csv next to script
    parser.add_argument("--input", help="Path to input CSV file (default: sample_cart_abandoners.csv next to script)")
    parser.add_argument("--output", help="Path to output .csv or .json (default: segments_output.csv next to script)")
    parser.add_argument("--min-seg-size", type=int, default=MIN_SEGMENT_SIZE, help="Minimum segment size")
    parser.add_argument("--max-seg-size", type=int, default=MAX_SEGMENT_SIZE, help="Maximum segment size")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS, help="Abandonment lookback window (days)")
    # Adaptive minimum size for small universes
    parser.add_argument(
        "--auto-min-size",
        action="store_true",
        help="Adapt minimum segment size to small universes using --min-seg-frac",
    )
    parser.add_argument(
        "--min-seg-frac",
        type=float,
        default=0.10,
        help="Fraction of universe for adaptive min size (used with --auto-min-size)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")
    return parser.parse_args()


def _default_paths() -> Tuple[Path, Path]:
    """Compute default input/output paths next to this script."""
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "sample_cart_abandoners.csv"
    default_output = script_dir / "segments_output.csv"
    return default_input, default_output


def load_data(input_path: str) -> pd.DataFrame:
    """Load CSV and parse date columns with basic type handling."""
    df = pd.read_csv(
        input_path,
        dtype={
            "user_id": str,
        },
        parse_dates=["cart_abandoned_date", "last_order_date"],
        dayfirst=False,
    )

    required_cols = {
        "user_id",
        "cart_abandoned_date",
        "last_order_date",
        "avg_order_value",
        "sessions_last_30d",
        "num_cart_items",
        "engagement_score",
        "profitability_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric dtypes and coerce errors to NaN
    for col in ["avg_order_value", "engagement_score", "profitability_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["sessions_last_30d", "num_cart_items"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float")

    # Drop duplicate user_id rows keeping the most recent abandoned date if any
    if df["user_id"].duplicated().any():
        df = (
            df.sort_values(["user_id", "cart_abandoned_date"], ascending=[True, False])
              .drop_duplicates(subset=["user_id"], keep="first")
        )

    return df


def filter_universe(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Filter users who abandoned a cart in the last `lookback_days` days."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=lookback_days)
    mask = (df["cart_abandoned_date"] >= cutoff) & (df["cart_abandoned_date"] <= today)
    uni = df.loc[mask].copy()
    logging.info("Universe filtered to last %d days: %d users", lookback_days, len(uni))
    return uni


def compute_thresholds(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute quantile thresholds for continuous features."""
    thresholds = {}

    def q(series: pd.Series, low=QUANTILE_LOW, high=QUANTILE_HIGH) -> Tuple[float, float]:
        s = series.dropna()
        return float(s.quantile(low)), float(s.quantile(high))

    aov_low, aov_high = q(df["avg_order_value"])
    eng_low, eng_high = q(df["engagement_score"])
    prof_low, prof_high = q(df["profitability_score"])
    ses_high = float(df["sessions_last_30d"].dropna().quantile(SESSIONS_HIGH_Q))
    items_high = float(df["num_cart_items"].dropna().quantile(ITEMS_HIGH_Q))

    thresholds["avg_order_value"] = {"low": aov_low, "high": aov_high}
    thresholds["engagement_score"] = {"low": eng_low, "high": eng_high}
    thresholds["profitability_score"] = {"low": prof_low, "high": prof_high}
    thresholds["sessions_last_30d"] = {"high": ses_high}
    thresholds["num_cart_items"] = {"high": items_high}

    logging.info(
        "Thresholds: AOV low=%.2f high=%.2f | ENG low=%.3f high=%.3f | PROF low=%.3f high=%.3f | SESS high=%.1f | ITEMS high=%.1f",
        aov_low, aov_high, eng_low, eng_high, prof_low, prof_high, ses_high, items_high,
    )
    return thresholds


def _stable_part_index(user_id: str, parts: int, seed: int = 0) -> int:
    """Deterministic partition index in [0, parts-1] using md5 hash of user_id and seed."""
    h = hashlib.md5(f"{seed}|{user_id}".encode("utf-8")).hexdigest()
    return int(h, 16) % parts


def assign_segments(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Assign MECE segments via ordered rules. Returns a DataFrame with 'segment' and 'rule' columns."""
    out = df.copy()

    # Helper columns for recency
    today = pd.Timestamp.today().normalize()
    out["recency_last_order_days"] = (today - out["last_order_date"]).dt.days
    out["recency_last_order_days"] = out["recency_last_order_days"].fillna(10_000)
    out["recency_abandon_days"] = (today - out["cart_abandoned_date"]).dt.days.clip(lower=0)

    # Rule definitions (ordered). Each returns a boolean mask over 'out'.
    aov = thresholds["avg_order_value"]
    eng = thresholds["engagement_score"]
    prof = thresholds["profitability_score"]
    ses = thresholds["sessions_last_30d"]
    items = thresholds["num_cart_items"]

    rules: List[Tuple[str, str, pd.Series]] = []

    # 1) High AOV Abandoners
    rules.append((
        "High AOV Abandoners",
        f"avg_order_value >= {aov['high']:.2f}",
        out["avg_order_value"] >= aov["high"],
    ))

    # 2) Large Cart, Highly Engaged
    rules.append((
        "Large Cart Engaged",
        f"num_cart_items >= {items['high']:.1f} AND engagement_score >= {eng['high']:.2f}",
        (out["num_cart_items"] >= items["high"]) & (out["engagement_score"] >= eng["high"]),
    ))

    # 3) Mid AOV, High Engagement
    rules.append((
        "Mid AOV Engaged",
        f"{aov['low']:.2f} <= avg_order_value < {aov['high']:.2f} AND engagement_score >= {eng['high']:.2f}",
        (out["avg_order_value"] >= aov["low"]) & (out["avg_order_value"] < aov["high"]) & (out["engagement_score"] >= eng["high"]),
    ))

    # 4) High Profit, Low Engagement
    rules.append((
        "High Profit Low Engagement",
        f"profitability_score >= {prof['high']:.2f} AND engagement_score < {eng['low']:.2f}",
        (out["profitability_score"] >= prof["high"]) & (out["engagement_score"] < eng["low"]),
    ))

    # 5) Highly Engaged Browsers
    rules.append((
        "Highly Engaged Browsers",
        f"sessions_last_30d >= {ses['high']:.1f} AND engagement_score >= {eng['high']:.2f}",
        (out["sessions_last_30d"] >= ses["high"]) & (out["engagement_score"] >= eng["high"]),
    ))

    # 6) Recent Buyers Abandoners
    rules.append((
        "Recent Buyers Abandoners",
        "recency_last_order_days <= 30",
        out["recency_last_order_days"] <= 30,
    ))

    # 7) Lapsed Abandoners
    rules.append((
        "Lapsed Abandoners",
        "last_order_date is null OR recency_last_order_days > 365",
        (out["last_order_date"].isna()) | (out["recency_last_order_days"] > 365),
    ))

    # 8) Low Value Price Sensitive
    rules.append((
        "Low Value Price Sensitive",
        f"avg_order_value < {aov['low']:.2f} AND profitability_score < {prof['low']:.2f}",
        (out["avg_order_value"] < aov["low"]) & (out["profitability_score"] < prof["low"]),
    ))

    # Assign sequentially to ensure mutual exclusivity
    # Initialize as object dtype to avoid dtype warnings when assigning strings
    out["segment"] = pd.Series([None] * len(out), dtype="object")
    out["rule"] = pd.Series([None] * len(out), dtype="object")
    unassigned = pd.Series(True, index=out.index)

    for seg_name, rule_text, mask in rules:
        assign_mask = unassigned & mask.fillna(False)
        out.loc[assign_mask, "segment"] = seg_name
        out.loc[assign_mask, "rule"] = rule_text
        unassigned = unassigned & (~assign_mask)

    # Remaining users go to Other
    remaining = out["segment"].isna()
    out.loc[remaining, "segment"] = "Other"
    out.loc[remaining, "rule"] = "Catch-all for remaining users"

    assigned_pct = 100.0 * (1.0 - unassigned.mean())
    logging.info("Initial assignment complete. Assigned: %.2f%%", assigned_pct)

    return out


def enforce_size_constraints(
    assigned: pd.DataFrame,
    min_size: int,
    max_size: int,
    allow_split: bool = SPLIT_OVERSIZED_SEGMENTS,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    """
    Enforce min/max segment sizes.

    - Segments smaller than min_size are merged into 'Other'.
    - Segments larger than max_size are split into parts if allow_split.

    Returns (final_assignments, merged_segments_summary_df, split_metadata_list)
    """
    df = assigned.copy()

    # Merge small segments (except 'Other')
    counts = df.groupby("segment").size().sort_values(ascending=False)
    small_segments = [s for s, n in counts.items() if n < min_size and s != "Other"]

    merged_rows = []
    if small_segments:
        logging.info("Merging small segments into Other: %s", ", ".join(small_segments))
        for s in small_segments:
            original_size = int(counts.get(s, 0))
            mask = df["segment"] == s
            df.loc[mask, "segment"] = "Other"
            df.loc[mask, "rule"] = df.loc[mask, "rule"].astype(str) + " | merged_into=Other"
            merged_rows.append({
                "segment": s,
                "original_size": original_size,
                "final_size": 0,
                "status": "merged",
                "merged_into": "Other",
            })

    merged_summary = pd.DataFrame(merged_rows)

    # Recompute counts after merges
    counts = df.groupby("segment").size().sort_values(ascending=False)

    # Split oversized segments (including Other if needed)
    split_meta: List[Dict[str, object]] = []
    if allow_split and max_size > 0:
        oversized = [(s, int(n)) for s, n in counts.items() if n > max_size]
        for seg, size in oversized:
            parts = ceil(size / max_size)
            logging.info("Splitting oversized segment '%s' (size=%d) into %d parts", seg, size, parts)
            mask = df["segment"] == seg
            idx = df.index[mask]
            # Deterministic partitioning by user_id hash
            part_idx = [
                _stable_part_index(str(df.at[i, "user_id"]), parts, seed=42) for i in idx
            ]
            df.loc[idx, "segment"] = [f"{seg} (part {p+1}/{parts})" for p in part_idx]
            split_meta.append({
                "segment": seg,
                "original_size": size,
                "parts": parts,
                "status": "split",
            })

    return df, merged_summary, split_meta


def compute_user_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user helper metrics used for segment aggregation."""
    out = df.copy()
    # recency_last_order_days already computed in assign_segments; ensure present
    if "recency_last_order_days" not in out.columns:
        today = pd.Timestamp.today().normalize()
        out["recency_last_order_days"] = (today - out["last_order_date"]).dt.days.fillna(10_000)
    if "recency_abandon_days" not in out.columns:
        today = pd.Timestamp.today().normalize()
        out["recency_abandon_days"] = (today - out["cart_abandoned_date"]).dt.days.clip(lower=0)

    # Conversion potential: engagement_score / (1 + recency_last_order_days)
    out["conversion_potential_u"] = out["engagement_score"] / (1.0 + out["recency_last_order_days"].astype(float))
    out["conversion_potential_u"] = out["conversion_potential_u"].fillna(0.0).clip(lower=0.0)

    # Strategic fit: combine profitability and recency (closer last order -> higher fit)
    # Normalize recency to [0,1] by capping at 365 days
    rec_norm = 1.0 - (out["recency_last_order_days"].clip(lower=0, upper=365) / 365.0)
    out["strategic_fit_u"] = 0.5 * out["profitability_score"].fillna(0.0) + 0.5 * rec_norm.fillna(0.0)
    out["strategic_fit_u"] = out["strategic_fit_u"].clip(lower=0.0, upper=1.0)

    return out


def aggregate_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-user metrics to segment-level metrics and compute overall score."""
    total = len(df)
    if total == 0:
        raise ValueError("No users in universe after filtering.")

    grp = df.groupby("segment")
    agg = grp.agg(
        size=("user_id", "count"),
        conversion_potential=("conversion_potential_u", "mean"),
        profitability=("profitability_score", "mean"),
        strategic_fit=("strategic_fit_u", "mean"),
    ).reset_index()

    # Normalized size
    agg["size_normalized"] = agg["size"] / float(total)

    # Normalize metrics to [0,1] across segments for fair weighting
    def minmax_norm(s: pd.Series) -> pd.Series:
        if s.nunique(dropna=True) <= 1:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    agg["conversion_potential_norm"] = minmax_norm(agg["conversion_potential"])  # already small -> rescale
    agg["profitability_norm"] = minmax_norm(agg["profitability"])  # already 0..1 but rescale across segs
    agg["strategic_fit_norm"] = minmax_norm(agg["strategic_fit"])  # 0..1 rescale across segs

    # size_normalized is already in [0,1]; keep as the normalized size metric
    w = WEIGHTS.copy()
    w_sum = sum(w.values())
    if not np.isclose(w_sum, 1.0):
        # Normalize weights to sum to 1
        w = {k: float(v) / w_sum for k, v in w.items()}

    agg["overall_score"] = (
        w["conversion_potential"] * agg["conversion_potential_norm"]
        + w["profitability"] * agg["profitability_norm"]
        + w["size"] * agg["size_normalized"]
        + w["strategic_fit"] * agg["strategic_fit_norm"]
    )

    return agg


def build_output_summary(
    segment_metrics: pd.DataFrame,
    merged_summary: pd.DataFrame,
    split_meta: List[Dict[str, object]],
    rules_by_segment: Dict[str, str],
) -> pd.DataFrame:
    """
    Build final summary including rules and status flags for merged/split segments.
    Only final (existing) segments will have sizes > 0. Merged segments are listed
    with final_size=0 and status='merged' to indicate the action.
    """
    seg_df = segment_metrics.copy()
    seg_df["status"] = "ok"
    seg_df["was_merged"] = False
    seg_df["was_split"] = False
    seg_df["merged_into"] = ""
    seg_df["original_size"] = seg_df["size"]

    # Apply split flags
    split_bases = {m["segment"]: m for m in split_meta}
    seg_df.loc[seg_df["segment"].str.contains("(part ", regex=False, na=False), "was_split"] = True
    # Add note of origin parts where applicable (best-effort identification)
    seg_df["split_origin"] = seg_df["segment"].str.extract(r"^(.*) \(part \d+/\d+\)$")[0].fillna("")

    # Append merged summary rows
    merged_rows = []
    if not merged_summary.empty:
        for _, r in merged_summary.iterrows():
            merged_rows.append({
                "segment": r["segment"],
                "size": 0,
                "size_normalized": 0.0,
                "conversion_potential": np.nan,
                "profitability": np.nan,
                "strategic_fit": np.nan,
                "conversion_potential_norm": np.nan,
                "profitability_norm": np.nan,
                "strategic_fit_norm": np.nan,
                "overall_score": np.nan,
                "status": "merged",
                "was_merged": True,
                "was_split": False,
                "merged_into": r.get("merged_into", "Other"),
                "original_size": r.get("original_size", np.nan),
                "split_origin": "",
            })

    full_df = pd.concat([seg_df, pd.DataFrame(merged_rows)], ignore_index=True, sort=False)

    # Attach human-readable rule definitions (best-effort for parts: inherit base rule)
    def resolve_rule(seg: str) -> str:
        if seg in rules_by_segment:
            return rules_by_segment[seg]
        # For parts, try base name
        if "(part" in seg:
            base = seg.split(" (part ")[0]
            return rules_by_segment.get(base, "")
        return ""

    full_df["rules"] = full_df["segment"].apply(resolve_rule)

    # Reorder columns for clarity
    columns = [
        "segment",
        "rules",
        "status",
        "was_merged",
        "was_split",
        "merged_into",
        "original_size",
        "size",
        "size_normalized",
        "conversion_potential",
        "profitability",
        "strategic_fit",
        "overall_score",
    ]
    existing_cols = [c for c in columns if c in full_df.columns]
    others = [c for c in full_df.columns if c not in existing_cols]
    full_df = full_df[existing_cols + others]

    return full_df


def write_output(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    ext = p.suffix.lower()

    def _write(target: Path):
        if ext == ".json":
            with open(target, "w", encoding="utf-8") as f:
                json.dump(json.loads(df.to_json(orient="records")), f, indent=2)
        else:
            # Default to CSV for unknown ext
            df.to_csv(target, index=False)

    try:
        _write(p)
        kind = "JSON" if ext == ".json" else "CSV"
        logging.info("Wrote %s to %s (%d segments)", kind, str(p), len(df))
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = p.with_name(f"{p.stem}_{ts}{p.suffix or '.csv'}")
        logging.warning(
            "Permission denied writing to %s (file may be open). Writing to %s instead.",
            str(p), str(fallback)
        )
        _write(fallback)
        kind = "JSON" if ext == ".json" else "CSV"
        logging.info("Wrote %s to %s (%d segments)", kind, str(fallback), len(df))


def main():
    args = parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        _setup_logging()

    # Resolve default input/output if not provided
    def_in, def_out = _default_paths()
    if not args.input:
        if def_in.exists():
            args.input = str(def_in)
            logging.info("No --input provided. Using default: %s", args.input)
        elif Path("sample_cart_abandoners.csv").exists():
            args.input = str(Path("sample_cart_abandoners.csv").resolve())
            logging.info("No --input provided. Using file in CWD: %s", args.input)
        else:
            raise FileNotFoundError(
                f"No --input provided and default file not found: {def_in}"
            )

    if not args.output:
        # Prefer to write next to script, otherwise CWD
        out_path = def_out
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Test we can open for write
            with open(out_path, "w", encoding="utf-8") as _:
                pass
            out_path.unlink(missing_ok=True)  # remove the probe file
            args.output = str(out_path)
            logging.info("No --output provided. Using default: %s", args.output)
        except Exception:
            args.output = str(Path("segments_output.csv").resolve())
            logging.info("No --output provided. Using CWD fallback: %s", args.output)

    logging.info("Loading data from %s", args.input)
    df = load_data(args.input)

    # Filter universe
    uni = filter_universe(df, args.lookback_days)
    if uni.empty:
        logging.warning("No users found in the last %d days. Exiting.", args.lookback_days)
        # Still output an empty file with headers for consistency
        write_output(pd.DataFrame(), args.output)
        return

    # Compute thresholds
    thresholds = compute_thresholds(uni)

    # Assign segments
    assigned = assign_segments(uni, thresholds)

    # Keep a mapping of base rules per segment for reporting
    # Note: 'Other' rule will be a generic catch-all
    rules_by_segment: Dict[str, str] = (
        assigned.groupby("segment")["rule"].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "").to_dict()
    )

    # Determine effective min segment size (adaptive for small universes if enabled)
    effective_min_size = args.min_seg_size
    if args.auto_min_size:
        adaptive = int(np.ceil(max(1.0, args.min_seg_frac * len(uni))))
        effective_min_size = min(args.min_seg_size, adaptive)
        logging.info(
            "Auto min size: base=%d, frac=%.2f, universe=%d -> effective_min=%d",
            args.min_seg_size, args.min_seg_frac, len(uni), effective_min_size,
        )

    # Enforce size constraints
    final_assignments, merged_summary, split_meta = enforce_size_constraints(
        assigned, effective_min_size, args.max_seg_size, allow_split=SPLIT_OVERSIZED_SEGMENTS
    )

    # Compute per-user metrics and aggregate to segments
    with_user_metrics = compute_user_metrics(final_assignments)
    segment_metrics = aggregate_segment_metrics(with_user_metrics)

    # Build output summary
    summary = build_output_summary(segment_metrics, merged_summary, split_meta, rules_by_segment)

    # Sort by overall score descending (NaNs last)
    summary = summary.sort_values(by=["overall_score"], ascending=[False], na_position="last").reset_index(drop=True)

    # Output
    write_output(summary, args.output)

    # Log a concise breakdown of final segment sizes
    size_breakdown = (
        with_user_metrics.groupby("segment").size().sort_values(ascending=False)
    )
    logging.info("Final segments and sizes:\n%s", size_breakdown.to_string())


if __name__ == "__main__":
    _setup_logging()
    main()
