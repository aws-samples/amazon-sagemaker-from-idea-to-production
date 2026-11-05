"""Generate a synthetic healthcare practitioner (HCP) prescribing dataset.

The data is entirely synthetic: no real HCPs, brands, or prescriptions. It is
shaped like a typical pharma commercial analytics problem: for each HCP and
each week, predict whether the HCP will write at least one prescription (Rx)
for "our brand" in the next 4 weeks, based on who they are, how they practice,
their recent prescribing history, and the outreach they received.

The generator has real, learnable structure built in:

- HCP profile (specialty, segment, practice type, formulary access) sets a
  baseline likelihood of prescribing the brand
- Outreach (rep calls, samples, emails, speaker programs) raises brand share,
  with carry-over from week to week and diminishing returns
- Many HCPs write few or no brand prescriptions ("lots of zeros")
- A competitor product launches in week ``COMPETITOR_LAUNCH_WEEK``. After the
  launch, brand share drops, most strongly in the specialties the competitor
  targets and unevenly across HCPs; the competitor also grows the market there,
  and the brand team responds with more emails and rep calls. The weeks after
  the launch are written to a separate file and used as "new production data"
  to demonstrate drift monitoring.

The output is deterministic for a given seed.

Usage:
    python generate_hcp_data.py --output-dir ../../data/synthetic-hcp
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
NUM_HCPS = 5000
NUM_SNAPSHOT_WEEKS = 26          # weekly snapshots written to the output
HISTORY_WEEKS = 26               # look-back window for prescribing history
HORIZON_WEEKS = 4                # prediction window: the next 4 weeks
COMPETITOR_LAUNCH_WEEK = 21      # snapshot week in which the competitor launches
FIRST_SNAPSHOT_DATE = "2025-01-06"  # a Monday

TRAINING_FILE = "hcp_weekly_snapshots.csv"
NEW_DATA_FILE = "hcp_weekly_snapshots_after_launch.csv"

# specialty: (share of HCPs, brand affinity, category Rx per 100 patient visits, targeted by competitor)
SPECIALTIES = {
    "Rheumatology":      (0.10,  0.7, 4.0, True),
    "Dermatology":       (0.12,  0.5, 3.5, True),
    "Gastroenterology":  (0.10,  0.4, 3.0, False),
    "Primary Care":      (0.34, -0.6, 1.2, False),
    "Internal Medicine": (0.16, -0.4, 1.5, False),
    "Allergy/Immunology":(0.06,  0.2, 2.5, True),
    "Nurse Practitioner":(0.08, -0.3, 1.5, False),
    "Other":             (0.04, -0.9, 0.8, False),
}
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
PRACTICE_TYPES = {"Academic": 0.10, "Hospital": 0.20, "Group practice": 0.45, "Solo practice": 0.25}
SEGMENTS = ["A", "B", "C", "D"]  # A = highest market volume
FORMULARY_ACCESS = {"Preferred": (0.35, 0.4), "Covered": (0.45, 0.0), "Restricted": (0.20, -0.7)}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_profiles(rng):
    n = NUM_HCPS
    spec_names = list(SPECIALTIES)
    spec_p = np.array([SPECIALTIES[s][0] for s in spec_names])
    specialty = rng.choice(spec_names, size=n, p=spec_p / spec_p.sum())

    practice = rng.choice(list(PRACTICE_TYPES), size=n, p=list(PRACTICE_TYPES.values()))
    region = rng.choice(REGIONS, size=n, p=[0.22, 0.24, 0.20, 0.14, 0.20])
    access_names = list(FORMULARY_ACCESS)
    access = rng.choice(access_names, size=n, p=[FORMULARY_ACCESS[a][0] for a in access_names])
    graduation_year = rng.integers(1972, 2020, size=n)

    # Weekly patient visits: lognormal, larger for hospital and academic practices
    practice_scale = pd.Series(practice).map(
        {"Academic": 1.4, "Hospital": 1.3, "Group practice": 1.0, "Solo practice": 0.7}
    ).to_numpy()
    patient_visits = np.round(rng.lognormal(mean=4.1, sigma=0.5, size=n) * practice_scale).astype(int)

    category_rate = pd.Series(specialty).map({s: v[2] for s, v in SPECIALTIES.items()}).to_numpy()
    market_lambda = patient_visits * category_rate / 100.0  # expected category Rx per week

    # Segment = market-volume quartile (A = top 25%)
    segment = pd.qcut(market_lambda, 4, labels=SEGMENTS[::-1]).astype(str)

    affinity = pd.Series(specialty).map({s: v[1] for s, v in SPECIALTIES.items()}).to_numpy()
    access_effect = pd.Series(access).map({a: v[1] for a, v in FORMULARY_ACCESS.items()}).to_numpy()
    practice_effect = pd.Series(practice).map(
        {"Academic": 0.3, "Hospital": 0.1, "Group practice": 0.0, "Solo practice": -0.2}
    ).to_numpy()
    # Younger HCPs adopt the (relatively new) brand slightly more often
    adoption_effect = (graduation_year - 1995) / 25.0 * 0.3
    hcp_random_effect = rng.normal(0.0, 0.7, size=n)
    base_logit = -2.4 + affinity + access_effect + practice_effect + adoption_effect + hcp_random_effect

    targeted = pd.Series(specialty).map({s: v[3] for s, v in SPECIALTIES.items()}).to_numpy()
    # How readily each HCP switches to the competitor after its launch (unobserved)
    switch_propensity = rng.beta(1.0, 1.0, size=n)

    # Outreach intensity: reps focus on high-volume, high-affinity HCPs (as in real life)
    segment_call_rate = pd.Series(segment).map({"A": 0.9, "B": 0.55, "C": 0.3, "D": 0.12}).to_numpy()
    call_rate = segment_call_rate * np.exp(0.25 * affinity) * rng.lognormal(0.0, 0.4, size=n)
    rep_access = rng.uniform(0.2, 1.0, size=n)  # some HCPs rarely see reps
    digital_affinity = rng.beta(2, 3, size=n)

    profiles = pd.DataFrame({
        "hcp_id": [f"HCP{100000 + i}" for i in range(n)],
        "specialty": specialty,
        "region": region,
        "practice_type": practice,
        "segment": segment,
        "formulary_access": access,
        "graduation_year": graduation_year,
        "patient_visits_weekly": patient_visits,
    })
    latent = {
        "market_lambda": market_lambda,
        "base_logit": base_logit,
        "targeted": targeted,
        "switch_propensity": switch_propensity,
        "call_rate": call_rate * rep_access,
        "digital_affinity": digital_affinity,
    }
    return profiles, latent


def simulate_weeks(rng, latent):
    """Simulate weekly outreach and prescriptions for every HCP."""
    n = NUM_HCPS
    total_weeks = HISTORY_WEEKS + NUM_SNAPSHOT_WEEKS + HORIZON_WEEKS
    # Absolute week index of the competitor launch in the simulation timeline
    launch_t = HISTORY_WEEKS + COMPETITOR_LAUNCH_WEEK - 1

    shape = (total_weeks, n)
    rep_calls = np.zeros(shape, dtype=int)
    samples = np.zeros(shape, dtype=int)
    emails = np.zeros(shape, dtype=int)
    speaker = np.zeros(shape, dtype=int)
    market_rx = np.zeros(shape, dtype=int)
    brand_rx = np.zeros(shape, dtype=int)

    adstock = np.zeros(n)
    week_noise = np.zeros(n)  # week-to-week ups and downs in each HCP's brand share
    for t in range(total_weeks):
        season = 1.0 + 0.08 * np.sin(2 * np.pi * t / 52.0)
        # After the launch: the competitor grows the market in its target specialties,
        # and the brand team responds with more emails and rep calls there
        ramp = min(1.0, max(0.0, (t - launch_t + 1) / 4.0))
        market_growth = 1.0 + ramp * np.where(latent["targeted"], 0.35, 0.05)
        call_boost = 1.0 + ramp * np.where(latent["targeted"], 0.6, 0.1)
        email_boost = 1.0 + ramp * 1.2

        rep_calls[t] = rng.poisson(latent["call_rate"] * call_boost)
        samples[t] = rng.binomial(rep_calls[t] * 3, 0.5)
        emails[t] = rng.poisson((0.3 + 1.5 * latent["digital_affinity"]) * email_boost)
        speaker[t] = rng.binomial(1, 0.004 + 0.02 * (latent["call_rate"] > 0.6))

        # Outreach carry-over ("adstock") with diminishing returns
        adstock = 0.75 * adstock + (1.0 * rep_calls[t] + 0.15 * samples[t]
                                    + 0.1 * emails[t] + 2.5 * speaker[t])
        outreach_effect = 0.9 * np.log1p(adstock)

        # Competitor takes share, ramping up over 4 weeks: strongest in targeted
        # specialties, and uneven across HCPs (some switch, some don't)
        competitor_effect = (ramp * latent["switch_propensity"]
                             * np.where(latent["targeted"], -4.5, -1.5))

        week_noise = 0.8 * week_noise + rng.normal(0.0, 0.45, size=n)
        brand_share = sigmoid(latent["base_logit"] + outreach_effect + competitor_effect + week_noise)
        market_rx[t] = rng.poisson(latent["market_lambda"] * season * market_growth)
        brand_rx[t] = rng.binomial(market_rx[t], brand_share)

    return dict(rep_calls=rep_calls, samples=samples, emails=emails, speaker=speaker,
                market_rx=market_rx, brand_rx=brand_rx)


def window_sum(x, end, length):
    """Sum of weeks (end - length, end] for every HCP. ``end`` is exclusive."""
    return x[end - length:end].sum(axis=0)


def build_snapshots(profiles, sim):
    frames = []
    dates = pd.date_range(FIRST_SNAPSHOT_DATE, periods=NUM_SNAPSHOT_WEEKS, freq="W-MON")
    for k in range(NUM_SNAPSHOT_WEEKS):
        t = HISTORY_WEEKS + k      # the snapshot is taken at the end of simulation week t-1
        snap = profiles.copy()
        snap.insert(1, "week_start_date", dates[k].strftime("%Y-%m-%d"))
        snap.insert(2, "week_index", k + 1)
        competitor = sim["market_rx"] - sim["brand_rx"]
        snap["brand_rx_last_4w"] = window_sum(sim["brand_rx"], t, 4)
        snap["brand_rx_last_13w"] = window_sum(sim["brand_rx"], t, 13)
        snap["brand_rx_last_26w"] = window_sum(sim["brand_rx"], t, 26)
        snap["competitor_rx_last_13w"] = window_sum(competitor, t, 13)
        snap["market_rx_last_13w"] = window_sum(sim["market_rx"], t, 13)
        snap["rep_calls_last_4w"] = window_sum(sim["rep_calls"], t, 4)
        snap["rep_calls_last_13w"] = window_sum(sim["rep_calls"], t, 13)
        snap["samples_last_4w"] = window_sum(sim["samples"], t, 4)
        snap["emails_last_4w"] = window_sum(sim["emails"], t, 4)
        snap["speaker_events_last_13w"] = window_sum(sim["speaker"], t, 13)
        # Target: brand prescriptions in the NEXT 4 weeks (not known at scoring time)
        snap["brand_rx_next_4w"] = sim["brand_rx"][t:t + HORIZON_WEEKS].sum(axis=0)
        frames.append(snap)
    return pd.concat(frames, ignore_index=True)


def add_missing_values(rng, df):
    """Real data is never complete: blank out a small share of a few fields."""
    for col, share in [("graduation_year", 0.03), ("patient_visits_weekly", 0.02)]:
        hcp_mask = rng.random(NUM_HCPS) < share
        missing_hcps = df["hcp_id"].unique()[hcp_mask]
        df.loc[df["hcp_id"].isin(missing_hcps), col] = np.nan
    return df


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    profiles, latent = make_profiles(rng)
    sim = simulate_weeks(rng, latent)
    df = build_snapshots(profiles, sim)
    df = add_missing_values(rng, df)
    before = df[df["week_index"] < COMPETITOR_LAUNCH_WEEK].reset_index(drop=True)
    after = df[df["week_index"] >= COMPETITOR_LAUNCH_WEEK].reset_index(drop=True)
    return before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    before, after = generate(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    before.to_csv(os.path.join(args.output_dir, TRAINING_FILE), index=False)
    after.to_csv(os.path.join(args.output_dir, NEW_DATA_FILE), index=False)
    for name, d in [(TRAINING_FILE, before), (NEW_DATA_FILE, after)]:
        print(f"{name}: {len(d):,} rows, weeks {d.week_index.min()}-{d.week_index.max()}, "
              f"share with brand Rx in next 4 weeks: {(d.brand_rx_next_4w > 0).mean():.1%}")


if __name__ == "__main__":
    main()
