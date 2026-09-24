#!/usr/bin/env python3
"""Regional SVCA dimensionality (report Fig. 1).

Runs `stringer19_region_comparison.run_tier1_dimensionality` (five sessions
per region, 1,000 neurons, 100-3,100 s, 1.25-s bins) with its figure written
to output/, saves the per-session records, and prints the between-region
Kruskal-Wallis tests quoted in the report.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.stats import kruskal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "pilot_analyses")]
import stringer19_region_comparison as rc

OUT = HERE / "output"  # generated figures, caches and tables (not tracked)
OUT.mkdir(exist_ok=True)


def main():
    rc.OUT_DIR = OUT
    records = rc.run_tier1_dimensionality()["records"]
    (OUT / "tier1_dimensionality_by_region.pdf").replace(OUT / "regional_dimensionality_panels.pdf")
    np.save(OUT / "regional_dimensionality_records.npy", np.array(records, dtype=object))
    for key in ("alpha", "dim50"):
        groups = [[r[key] for r in records if r["region"] == g] for g in rc.REGION_COLORS]
        print(key, {g: (round(np.mean(v), 2), round(np.std(v, ddof=1), 2))
                    for g, v in zip(rc.REGION_COLORS, groups)},
              f"Kruskal-Wallis p={kruskal(*groups).pvalue:.3f}")


if __name__ == "__main__":
    main()
