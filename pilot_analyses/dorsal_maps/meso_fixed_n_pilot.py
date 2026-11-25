"""One-draw, matched-cohort mesoscope pilot at 50, 100 and 200 neurons."""
import zlib

from iblatlas.regions import BrainRegions
import numpy as np
import pandas as pd

from decode_v2 import OUT_V2, TARGETS, apply_coverage, decode_sets
from region_variance import load, region_permutation
from sessions import load_session, session_files

SIZES = (50, 100, 200)
STEM = "meso_fixed_n_pilot"
KEYS = ["session", "region", "target", "decoder_neurons"]


def matched_pairs():
    meso = load()[0]["meso"]
    keys = ["session", "animal", "region", "n_neurons", "n_trials"]
    coverage = meso.groupby(keys).target.nunique()
    pairs = coverage[coverage.eq(len(TARGETS))].reset_index().drop(columns="target")
    return pairs.loc[pairs.n_neurons.ge(max(SIZES))].sort_values(["session", "region"])


def sample_session(session_id, features, pairs, brain_regions):
    session = apply_coverage(session_id, load_session(features), "meso")
    region = brain_regions.id2acronym(session["atlas_id"], mapping="Beryl")
    pools = []
    for row in pairs.itertuples(index=False):
        indices = np.flatnonzero(region == row.region)
        if len(indices) != row.n_neurons or len(session["choice"]) != row.n_trials:
            raise ValueError(f"Feature counts changed since Figure 4a for {session_id}/{row.region}")
        rng = np.random.default_rng(zlib.crc32(f"{session_id}/{row.region}".encode()))
        pools.append(rng.permutation(indices)[:max(SIZES)])

    return session, pools


def decode_session(session_id, features, pairs, brain_regions):
    session, pools = sample_session(session_id, features, pairs, brain_regions)
    blocks = []
    columns = ["session", "animal", "region"]
    for size in SIZES:
        decoding = decode_sets(session, [pool[:size] for pool in pools])
        for target, values in decoding.items():
            blocks.append(pairs[columns].assign(target=target, decoder_neurons=size, score=values))
    return pd.concat(blocks, ignore_index=True)


def scores_for(pairs):
    files = session_files("meso")
    regions = BrainRegions()
    checkpoints = OUT_V2 / f"_{STEM}_sessions"
    checkpoints.mkdir(exist_ok=True)
    frames = []
    for i, (session_id, session_pairs) in enumerate(pairs.groupby("session"), 1):
        path = checkpoints / f"{session_id}.pqt"
        cached = path.exists()
        score = (pd.read_parquet(path) if cached
                 else decode_session(session_id, files[session_id], session_pairs, regions))
        expected = pd.MultiIndex.from_product(
            [[session_id], session_pairs.region, TARGETS, SIZES], names=KEYS)
        actual = pd.MultiIndex.from_frame(score[KEYS])
        if actual.has_duplicates or set(expected) != set(actual) or not np.isfinite(score.score).all():
            raise ValueError(f"Missing, invalid or mismatched pilot scores for {session_id}")
        if not cached:
            tmp = path.with_suffix(".tmp")
            score.to_parquet(tmp, index=False)
            tmp.replace(path)
        frames.append(score)
        print(f"{i}/{pairs.session.nunique()} {session_id}: {len(session_pairs)} pairs", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarise(scores):
    rows = []
    for (target, size), group in scores.groupby(["target", "decoder_neurons"], sort=False):
        # Neuron count is fixed within each fit size. Session is the only nuisance term.
        region = region_permutation(group.rename(columns={"score": "all"}).assign(logN=0), n_perm=0)
        rows.append(dict(target=target, decoder_neurons=size, mean_score=group.score.mean(),
                         median_score=group.score.median(), region_partial_r2=region["region_partial_r2"],
                         n_pairs=len(group), n_sessions=region["region_sessions"],
                         n_mice=region["region_mice"], n_regions=region["region_labels"]))
    return pd.DataFrame(rows)


def main():
    pairs = matched_pairs()
    scores = scores_for(pairs)
    summary = summarise(scores)
    scores.to_parquet(OUT_V2 / f"{STEM}_scores.pqt", index=False)
    summary.to_csv(OUT_V2 / f"{STEM}_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
