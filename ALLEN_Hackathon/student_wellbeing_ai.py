"""
===============================================================================
STUDENT WELLBEING EARLY-WARNING AI
VERSION 3.0 — TRAJECTORY ENGINE
===============================================================================

Purpose
-------
A privacy-conscious behavioral early-warning prototype for school environments.

The system:
    1. Generates a synthetic 9-week student dataset.
    2. Learns each student's historical behavioral baseline.
    3. Calculates weekly feature-level deviations.
    4. Calculates weekly personalized risk.
    5. Calculates population-level anomaly risk.
    6. Calculates multi-signal corroboration.
    7. Produces a weekly 0-100 behavioral wellbeing risk score.
    8. Calculates rolling risk.
    9. Calculates risk velocity / slope.
   10. Calculates risk acceleration.
   11. Detects the first meaningful behavioral deviation.
   12. Identifies the earliest deviating signal.
   13. Calculates early-warning lead time.
   14. Generates replay-ready CSV files.
   15. Generates current student risk cards.
"""

from __future__ import annotations

import os
import json
from typing import Tuple

import numpy as np
import pandas as pd

from risk_score_calculation import (
    BASELINE_WEEKS,
    STRONG_Z_THRESHOLD,
    Z_THRESHOLD,
    CONVERGENCE_SIGNAL_COUNT,
    FEATURES,
    FEATURE_CONFIG,
    HIGH_THRESHOLD,
    MODERATE_THRESHOLD,
    RANDOM_SEED,
    VERY_HIGH_THRESHOLD,
    WATCH_THRESHOLD,
    build_baselines,
    calculate_feature_risks,
    calculate_final_weekly_risk,
    calculate_population_anomaly,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_STUDENTS = 50
N_WEEKS = 9

# Rolling risk window.
ROLLING_WINDOW = 3
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def trajectory_state(
    rolling_risk: float,
    velocity: float,
    acceleration: float,
) -> str:

    if rolling_risk >= HIGH_THRESHOLD and velocity >= 8:
        return "RAPIDLY ESCALATING"

    if rolling_risk >= MODERATE_THRESHOLD and velocity >= 5:
        return "ESCALATING"

    if velocity >= 3 or (
        acceleration >= 2 and rolling_risk >= WATCH_THRESHOLD
    ):
        return "EMERGING"

    return "STABLE"


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_dataset(
    n_students: int = N_STUDENTS,
    n_weeks: int = N_WEEKS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []

    for student_index in range(1, n_students + 1):

        student_id = f"STU-{student_index:03d}"

        # Student-specific behavioral characteristics.
        participation_base = rng.normal(72, 7)
        arrival_base = rng.normal(8.0, 0.15)
        departure_base = rng.normal(0.4, 0.15)
        extracurricular_base = rng.normal(4.0, 0.8)
        library_base = rng.normal(3.0, 0.7)
        lms_base = rng.normal(48, 8)
        help_base = rng.normal(1.5, 0.5)
        meal_base = rng.normal(4.0, 0.5)
        transport_base = rng.normal(0.4, 0.15)
        schedule_base = rng.normal(0.3, 0.1)
        digital_base = rng.normal(0.5, 0.15)

        # Clamp sensible ranges.
        participation_base = np.clip(participation_base, 45, 95)
        arrival_base = np.clip(arrival_base, 7.4, 8.8)
        departure_base = np.clip(departure_base, 0, 1.5)
        extracurricular_base = np.clip(extracurricular_base, 1, 6)
        library_base = np.clip(library_base, 1, 6)
        lms_base = np.clip(lms_base, 20, 80)
        help_base = np.clip(help_base, 0, 4)
        meal_base = np.clip(meal_base, 2, 5)
        transport_base = np.clip(transport_base, 0, 1.5)
        schedule_base = np.clip(schedule_base, 0, 1)
        digital_base = np.clip(digital_base, 0, 1)

        # -------------------------------------------------------------
        # Synthetic hidden behavioral patterns.
        # These create realistic trajectories for demonstration/testing.
        # -------------------------------------------------------------

        pattern = "stable"

        if student_index in [7, 19, 31]:
            pattern = "gradual_escalation"

        elif student_index in [12, 28]:
            pattern = "sudden_change"

        elif student_index in [4, 42]:
            pattern = "academic_engagement_decline"

        elif student_index in [15, 36]:
            pattern = "social_withdrawal"

        elif student_index in [23]:
            pattern = "transport_schedule_shift"

        elif student_index in [9, 47]:
            pattern = "temporary_spike"

        for week in range(1, n_weeks + 1):

            # Natural student-to-student variation.
            participation = participation_base + rng.normal(0, 4)
            arrival = arrival_base + rng.normal(0, 0.08)
            departure = departure_base + rng.normal(0, 0.15)
            extracurricular = extracurricular_base + rng.normal(0, 0.4)
            library = library_base + rng.normal(0, 0.35)
            lms = lms_base + rng.normal(0, 4)
            help_requests = help_base + rng.normal(0, 0.3)
            meal = meal_base + rng.normal(0, 0.25)
            transport = transport_base + rng.normal(0, 0.12)
            schedule = schedule_base + rng.normal(0, 0.08)
            digital = digital_base + rng.normal(0, 0.08)

            # ---------------------------------------------------------
            # Inject patterns.
            # ---------------------------------------------------------

            if pattern == "gradual_escalation" and week >= 5:

                intensity = week - 4

                participation -= 4.0 * intensity
                arrival += 0.06 * intensity
                departure += 0.18 * intensity
                extracurricular -= 0.30 * intensity
                lms -= 2.5 * intensity
                meal -= 0.18 * intensity
                schedule += 0.08 * intensity
                digital += 0.06 * intensity

            elif pattern == "sudden_change" and week >= 7:

                participation -= 15
                arrival += 0.35
                departure += 0.7
                extracurricular -= 1.8
                lms -= 15
                meal -= 0.8
                schedule += 0.45
                digital += 0.35

            elif pattern == "academic_engagement_decline" and week >= 6:

                intensity = week - 5

                participation -= 2.0 * intensity
                library -= 0.5 * intensity
                lms -= 5.0 * intensity
                help_requests += 0.7 * intensity
                digital += 0.05 * intensity

            elif pattern == "social_withdrawal" and week >= 5:

                intensity = week - 4

                extracurricular -= 0.75 * intensity
                library -= 0.15 * intensity
                meal -= 0.20 * intensity
                participation -= 1.5 * intensity

            elif pattern == "transport_schedule_shift" and week >= 6:

                intensity = week - 5

                arrival += 0.15 * intensity
                transport += 0.35 * intensity
                schedule += 0.18 * intensity
                departure += 0.1 * intensity

            elif pattern == "temporary_spike":

                if week in [6, 7]:

                    participation -= 14
                    arrival += 0.4
                    departure += 0.8
                    extracurricular -= 1.6
                    lms -= 12
                    meal -= 0.7
                    schedule += 0.5
                    digital += 0.4

            # Keep everything physically/logically sensible.
            participation = np.clip(participation, 0, 100)
            arrival = np.clip(arrival, 6.5, 10)
            departure = np.clip(departure, 0, 5)
            extracurricular = np.clip(extracurricular, 0, 10)
            library = np.clip(library, 0, 10)
            lms = np.clip(lms, 0, 120)
            help_requests = np.clip(help_requests, 0, 10)
            meal = np.clip(meal, 0, 7)
            transport = np.clip(transport, 0, 5)
            schedule = np.clip(schedule, 0, 5)
            digital = np.clip(digital, 0, 5)

            rows.append({
                "student_id": student_id,
                "week": week,
                "class_participation": participation,
                "arrival_irregularity": arrival,
                "early_departures": departure,
                "extracurricular_activity": extracurricular,
                "library_resource_usage": library,
                "lms_session_duration": lms,
                "academic_help_requests": help_requests,
                "meal_usage": meal,
                "transport_irregularity": transport,
                "schedule_changes": schedule,
                "digital_timing_shift": digital,
                "synthetic_pattern": pattern,
            })

    return pd.DataFrame(rows)


# =============================================================================
# TRAJECTORY ENGINE
# =============================================================================

def calculate_linear_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if len(x) < 2:
        return 0.0

    return float(np.polyfit(x, y, 1)[0])


def calculate_trajectory_for_student(
    group: pd.DataFrame,
) -> pd.DataFrame:

    group = group.sort_values("week").copy()

    risks = group["weekly_risk"].to_numpy(dtype=float)
    weeks = group["week"].to_numpy(dtype=float)

    rolling_risks = []

    velocities = []
    accelerations = []

    previous_velocity = 0.0

    for i in range(len(group)):

        start = max(0, i - ROLLING_WINDOW + 1)

        local_weeks = weeks[start:i + 1]
        local_risks = risks[start:i + 1]

        rolling = float(np.mean(local_risks))

        rolling_risks.append(rolling)

        # Current slope.
        velocity = calculate_linear_slope(
            local_weeks,
            local_risks,
        )

        velocities.append(velocity)

        # Acceleration = change in slope.
        acceleration = velocity - previous_velocity

        accelerations.append(acceleration)

        previous_velocity = velocity

    group["rolling_risk_3w"] = rolling_risks
    group["risk_velocity"] = velocities
    group["risk_acceleration"] = accelerations

    group["trajectory_state"] = [
        trajectory_state(
            rolling_risk=rolling_risks[i],
            velocity=velocities[i],
            acceleration=accelerations[i],
        )
        for i in range(len(group))
    ]

    return group


def add_trajectory_engine(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:

    pieces = []

    for student_id, group in risk_df.groupby("student_id"):

        trajectory = calculate_trajectory_for_student(group)

        pieces.append(trajectory)

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    return result.sort_values(
        ["student_id", "week"]
    ).reset_index(drop=True)


# =============================================================================
# FIRST MEANINGFUL DEVIATION
# =============================================================================

def detect_first_meaningful_deviation(
    group: pd.DataFrame,
    baseline_weeks: int = BASELINE_WEEKS,
) -> Tuple[int, str, str]:

    group = group.sort_values("week").copy()

    # We only allow deviation detection AFTER the historical
    # baseline is complete.
    candidate = group[
        group["week"] > baseline_weeks
    ].copy()

    if candidate.empty:
        return -1, "NONE", "No post-baseline weeks available"

    previous_signals = set()

    for _, row in candidate.iterrows():

        week = int(row["week"])

        signals = [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= Z_THRESHOLD
        ]

        strong_signals = [
            feature
            for feature in FEATURES
            if row[f"{feature}_z"] >= STRONG_Z_THRESHOLD
        ]

        # -------------------------------------------------------------
        # Criterion A:
        # At least two independent signals deviate together.
        # -------------------------------------------------------------

        if len(signals) >= CONVERGENCE_SIGNAL_COUNT:

            earliest = min(
                signals,
                key=lambda f: row[f"{f}_z"]
            )

            # Actually select highest-z signal.
            earliest = max(
                signals,
                key=lambda f: row[f"{f}_z"]
            )

            return (
                week,
                earliest,
                "MULTI_SIGNAL_CONVERGENCE"
            )

        # -------------------------------------------------------------
        # Criterion B:
        # One strong signal persists across consecutive weeks.
        # -------------------------------------------------------------

        if len(strong_signals) >= 1:

            persistent = (
                set(strong_signals)
                & previous_signals
            )

            if persistent:

                strongest = max(
                    persistent,
                    key=lambda f: row[f"{f}_z"]
                )

                return (
                    week - 1,
                    strongest,
                    "PERSISTENT_SINGLE_SIGNAL"
                )

        previous_signals = set(strong_signals)

    return (
        -1,
        "NONE",
        "No statistically meaningful post-baseline deviation detected"
    )


# =============================================================================
# EXPLANATION ENGINE
# =============================================================================

def generate_explanation(
    row: pd.Series,
) -> str:

    contributors = []

    for feature in FEATURES:

        z = row[f"{feature}_z"]

        if z >= Z_THRESHOLD:

            label = FEATURE_CONFIG[feature]["label"]

            contributors.append(
                f"{label} (z={z:.1f})"
            )

    contributors.sort(
        key=lambda x: float(
            x.split("z=")[1].rstrip(")")
        ),
        reverse=True,
    )

    if not contributors:

        return "No major independent behavioral deviations detected."

    top = contributors[:3]

    return (
        "Main deviations: "
        + "; ".join(top)
        + "."
    )


# =============================================================================
# STUDENT SUMMARY CARDS
# =============================================================================

def build_student_cards(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:

    cards = []

    for student_id, group in trajectory_df.groupby("student_id"):

        group = group.sort_values("week")

        current = group.iloc[-1]

        first_week, first_signal, detection_type = (
            detect_first_meaningful_deviation(group)
        )

        current_week = int(current["week"])

        if first_week == -1:

            lead_time = 0

        else:

            lead_time = max(
                0,
                current_week - first_week
            )

        weeks_watch = int(
            np.sum(
                group["weekly_risk"] >= WATCH_THRESHOLD
            )
        )

        weeks_moderate = int(
            np.sum(
                group["weekly_risk"] >= MODERATE_THRESHOLD
            )
        )

        weeks_high = int(
            np.sum(
                group["weekly_risk"] >= HIGH_THRESHOLD
            )
        )

        # Highest-contribution signals in current week.
        contributors = []

        for feature in FEATURES:

            contributors.append(
                (
                    feature,
                    float(current[f"{feature}_risk"]),
                )
            )

        contributors.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        top_contributors = [
            FEATURE_CONFIG[f]["label"]
            for f, _ in contributors[:3]
        ]

        # Confidence is confidence in behavioral-pattern evidence,
        # NOT confidence in a diagnosis.
        signal_count = int(
            current["meaningful_signal_count"]
        )

        anomaly = float(
            current["population_anomaly"]
        )

        if signal_count >= 3 and anomaly >= 60:
            confidence = "HIGH"

        elif signal_count >= 2 or anomaly >= 50:
            confidence = "MEDIUM"

        else:
            confidence = "LOW"

        cards.append({

            "student_id": student_id,

            "current_week": current_week,

            "current_risk_score": round(
                float(current["weekly_risk"]),
                2
            ),

            "current_severity": current["severity"],

            "trajectory_state": current[
                "trajectory_state"
            ],

            "risk_velocity": round(
                float(current["risk_velocity"]),
                3
            ),

            "risk_acceleration": round(
                float(current["risk_acceleration"]),
                3
            ),

            "first_deviation_week": first_week,

            "earliest_signal": (
                FEATURE_CONFIG[first_signal]["label"]
                if first_signal in FEATURE_CONFIG
                else "NONE"
            ),

            "deviation_detection_type": detection_type,

            "early_warning_lead_time_weeks": lead_time,

            "weeks_above_watch": weeks_watch,

            "weeks_above_moderate": weeks_moderate,

            "weeks_above_high": weeks_high,

            "meaningful_signal_count": signal_count,

            "population_anomaly": round(
                anomaly,
                2
            ),

            "confidence": confidence,

            "top_contributors": (
                " | ".join(top_contributors)
            ),

            "explanation": generate_explanation(
                current
            ),

        })

    cards_df = pd.DataFrame(cards)

    cards_df = cards_df.sort_values(
        "current_risk_score",
        ascending=False,
    ).reset_index(drop=True)

    cards_df.insert(
        0,
        "rank",
        np.arange(1, len(cards_df) + 1),
    )

    return cards_df


# =============================================================================
# REPLAY DATA
# =============================================================================

def build_replay_dataset(
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> pd.DataFrame:

    replay = trajectory_df.copy()

    # Map first deviation information back onto every week.
    card_lookup = cards_df.set_index("student_id")

    replay["first_deviation_week"] = replay[
        "student_id"
    ].map(
        card_lookup["first_deviation_week"]
    )

    replay["earliest_signal"] = replay[
        "student_id"
    ].map(
        card_lookup["earliest_signal"]
    )

    replay["lead_time_if_current"] = replay.apply(
        lambda row: (
            max(
                0,
                int(row["week"])
                - int(row["first_deviation_week"])
            )
            if int(row["first_deviation_week"]) > 0
            else 0
        ),
        axis=1,
    )

    # Replay-friendly ordering.
    columns = [
        "student_id",
        "week",

        "weekly_risk",
        "rolling_risk_3w",
        "severity",

        "risk_velocity",
        "risk_acceleration",
        "trajectory_state",

        "personalized_risk",
        "population_anomaly",
        "corroboration",

        "meaningful_signal_count",
        "strong_signal_count",
        "meaningful_signals",

        "first_deviation_week",
        "earliest_signal",
        "lead_time_if_current",
    ]

    return replay[columns].sort_values(
        ["student_id", "week"]
    )


# =============================================================================
# SIGNAL HEATMAP DATA
# =============================================================================

def build_signal_heatmap(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in trajectory_df.iterrows():

        for feature in FEATURES:

            rows.append({

                "student_id": row["student_id"],

                "week": row["week"],

                "signal": feature,

                "signal_label": FEATURE_CONFIG[
                    feature
                ]["label"],

                "z_score": row[
                    f"{feature}_z"
                ],

                "risk": row[
                    f"{feature}_risk"
                ],

                "meaningful_deviation": (
                    row[f"{feature}_z"]
                    >= Z_THRESHOLD
                ),

            })

    return pd.DataFrame(rows)


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_outputs(
    original_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
):

    expected_rows = N_STUDENTS * N_WEEKS

    assert len(original_df) == expected_rows, (
        "Unexpected number of raw data rows."
    )

    assert len(trajectory_df) == expected_rows, (
        "Unexpected number of trajectory rows."
    )

    assert len(cards_df) == N_STUDENTS, (
        "Unexpected number of student cards."
    )

    assert (
        trajectory_df["weekly_risk"].between(0, 100).all()
    ), "Weekly risk outside 0-100."

    assert (
        trajectory_df["rolling_risk_3w"]
        .between(0, 100)
        .all()
    ), "Rolling risk outside 0-100."

    assert (
        trajectory_df["population_anomaly"]
        .between(0, 100)
        .all()
    ), "Population anomaly outside 0-100."

    # Every student must have exactly 9 weeks.
    week_counts = trajectory_df.groupby(
        "student_id"
    )["week"].nunique()

    assert (
        week_counts == N_WEEKS
    ).all(), "At least one student does not have all weeks."

    print("\nVALIDATION PASSED")
    print("-----------------")
    print(f"Students: {N_STUDENTS}")
    print(f"Weeks/student: {N_WEEKS}")
    print(f"Trajectory rows: {len(trajectory_df)}")
    print("Risk range: 0-100")
    print("All trajectory records complete.")


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    raw_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    heatmap_df: pd.DataFrame,
    output_dir: str = "wellbeing_outputs",
):

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(
        output_dir,
        "synthetic_student_behavior_9_weeks.csv",
    )

    trajectory_path = os.path.join(
        output_dir,
        "student_trajectory_engine.csv",
    )

    cards_path = os.path.join(
        output_dir,
        "student_risk_cards.csv",
    )

    replay_path = os.path.join(
        output_dir,
        "early_warning_replay.csv",
    )

    heatmap_path = os.path.join(
        output_dir,
        "signal_heatmap_data.csv",
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    trajectory_df.to_csv(
        trajectory_path,
        index=False,
    )

    cards_df.to_csv(
        cards_path,
        index=False,
    )

    replay_df.to_csv(
        replay_path,
        index=False,
    )

    heatmap_df.to_csv(
        heatmap_path,
        index=False,
    )

    # JSON version for easy dashboard/API integration.
    json_path = os.path.join(
        output_dir,
        "student_risk_cards.json",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cards_df.to_dict(orient="records"),
            f,
            indent=2,
        )

    return {
        "raw": raw_path,
        "trajectory": trajectory_path,
        "cards": cards_path,
        "replay": replay_path,
        "heatmap": heatmap_path,
        "json": json_path,
    }


# =============================================================================
# TERMINAL REPORT
# =============================================================================

def print_final_report(
    cards_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
):

    print("\n")
    print("=" * 80)
    print("STUDENT WELLBEING EARLY-WARNING AI")
    print("TRAJECTORY ENGINE — FINAL REPORT")
    print("=" * 80)

    print(
        f"\nStudents analysed : {cards_df['student_id'].nunique()}"
    )

    print(
        f"Weeks analysed   : {trajectory_df['week'].nunique()}"
    )

    print(
        f"Total trajectories: {len(trajectory_df)}"
    )

    print("\nTOP CURRENT RISK TRAJECTORIES")
    print("-" * 80)

    display_columns = [
        "rank",
        "student_id",
        "current_risk_score",
        "current_severity",
        "trajectory_state",
        "risk_velocity",
        "risk_acceleration",
        "first_deviation_week",
        "early_warning_lead_time_weeks",
        "confidence",
    ]

    print(
        cards_df[
            display_columns
        ].head(10).to_string(index=False)
    )

    print("\n")
    print("TRAJECTORY REPLAY EXAMPLE")
    print("-" * 80)

    # Pick the highest current-risk student.
    top_student = cards_df.iloc[0]["student_id"]

    student_history = trajectory_df[
        trajectory_df["student_id"] == top_student
    ].sort_values("week")

    replay_columns = [
        "week",
        "weekly_risk",
        "rolling_risk_3w",
        "risk_velocity",
        "risk_acceleration",
        "trajectory_state",
        "meaningful_signal_count",
    ]

    print(
        f"\nStudent: {top_student}\n"
    )

    print(
        student_history[
            replay_columns
        ].to_string(index=False)
    )

    print("\n")
    print("PRIVACY / SAFETY NOTE")
    print("-" * 80)

    print(
        "This system detects unusual behavioral change patterns. "
        "It does not diagnose stress, depression, or any other condition. "
        "High-risk records require human review and contextual verification."
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    print("=" * 80)
    print("STUDENT WELLBEING EARLY-WARNING AI")
    print("VERSION 3.0 — TRAJECTORY ENGINE")
    print("=" * 80)

    print("\n[1/9] Generating synthetic 9-week student dataset...")

    raw_df = generate_synthetic_dataset()

    print(
        f"       Generated {len(raw_df)} records "
        f"for {N_STUDENTS} students."
    )

    print("\n[2/9] Building individual historical baselines...")

    baselines = build_baselines(
        raw_df,
        BASELINE_WEEKS,
    )

    print(
        f"       Baseline period: weeks 1-{BASELINE_WEEKS}"
    )

    print("\n[3/9] Calculating weekly feature-level deviations...")

    _, risk_df = calculate_feature_risks(
        raw_df,
        baselines,
    )

    print(
        f"       Calculated {len(FEATURES)} independent signals."
    )

    print("\n[4/9] Calculating population-level anomalies...")

    risk_df = calculate_population_anomaly(
        raw_df,
        risk_df,
    )

    print(
        "       Isolation Forest population anomaly model complete."
    )

    print("\n[5/9] Calculating final weekly behavioral risk...")

    risk_df = calculate_final_weekly_risk(
        risk_df,
    )

    print(
        "       Weekly 0-100 scores generated."
    )

    print("\n[6/9] Running trajectory engine...")

    trajectory_df = add_trajectory_engine(
        risk_df,
    )

    print(
        "       Rolling risk + velocity + acceleration calculated."
    )

    print("\n[7/9] Detecting early deviations and lead time...")

    cards_df = build_student_cards(
        trajectory_df,
    )

    print(
        "       First-deviation analysis complete."
    )

    print("\n[8/9] Generating replay datasets...")

    replay_df = build_replay_dataset(
        trajectory_df,
        cards_df,
    )

    heatmap_df = build_signal_heatmap(
        trajectory_df,
    )

    print(
        "       Replay and signal-level datasets generated."
    )

    print("\n[9/9] Validating and saving outputs...")

    validate_outputs(
        raw_df,
        trajectory_df,
        cards_df,
    )

    paths = save_outputs(
        raw_df,
        trajectory_df,
        cards_df,
        replay_df,
        heatmap_df,
    )

    print_final_report(
        cards_df,
        trajectory_df,
    )

    print("\n")
    print("=" * 80)
    print("FILES GENERATED")
    print("=" * 80)

    for name, path in paths.items():

        print(
            f"{name:12s} -> {path}"
        )

    print("\n")
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)

    print(
        "\nThe trajectory engine is now part of the core Python pipeline."
    )

    print(
        "Every student has a 9-week behavioral trajectory, "
        "rolling risk, velocity, acceleration, deviation detection, "
        "and early-warning lead time."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()