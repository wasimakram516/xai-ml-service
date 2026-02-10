import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis

BASE_PATH = "data/oulad/"
KEY_COLS = ["id_student", "code_module", "code_presentation"]
EARLY_DAYS = 28


# ======================================================
# LOAD DATA
# ======================================================
def load_oulad():
    student_info = pd.read_csv(BASE_PATH + "studentInfo.csv")
    registration = pd.read_csv(BASE_PATH + "studentRegistration.csv")
    assessments = pd.read_csv(BASE_PATH + "studentAssessment.csv")
    vle = pd.read_csv(BASE_PATH + "studentVle.csv")
    vle_meta = pd.read_csv(BASE_PATH + "vle.csv")
    assessments_table = pd.read_csv(BASE_PATH + "assessments.csv")
    courses = pd.read_csv(BASE_PATH + "courses.csv")
    return student_info, registration, assessments, vle, vle_meta, assessments_table, courses


def _filter_early(df, early_only, day_col="date", cutoff=EARLY_DAYS):
    if not early_only:
        return df
    if day_col not in df.columns:
        return df
    return df[df[day_col].fillna(10**9) <= cutoff].copy()


def _safe_skew(x):
    x = np.asarray(x)
    if x.size < 3 or np.nanstd(x) < 1e-9:
        return 0.0
    return float(skew(x, nan_policy="omit"))


def _safe_kurt(x):
    x = np.asarray(x)
    if x.size < 4 or np.nanstd(x) < 1e-9:
        return 0.0
    return float(kurtosis(x, nan_policy="omit"))


# ======================================================
# WEEKLY CLICKS
# ======================================================
def add_academic_weeks(vle, vle_meta, early_only=False):
    df = vle.merge(vle_meta[["id_site", "week_from"]], on="id_site", how="left")
    df = _filter_early(df, early_only, day_col="date")
    df["week"] = df["week_from"].fillna(0).astype(int)

    weekly = df.groupby(KEY_COLS + ["week"])["sum_click"].sum().unstack(fill_value=0)
    weekly.columns = [f"week_{int(c)}_clicks" for c in weekly.columns]
    return weekly.reset_index()


# ======================================================
# ACTIVITY TYPES + ENTROPY
# ======================================================
def add_activity_type_clicks(vle, vle_meta, early_only=False):
    merged = vle.merge(vle_meta[["id_site", "activity_type"]], on="id_site", how="left")
    merged = _filter_early(merged, early_only, day_col="date")
    merged["activity_type"] = merged["activity_type"].fillna("unknown")

    type_counts = merged["activity_type"].value_counts()
    rarity = np.log1p(len(merged) / (type_counts + 1))
    merged["weighted_clicks"] = merged["sum_click"] * merged["activity_type"].map(rarity)

    agg = merged.groupby(KEY_COLS + ["activity_type"])["weighted_clicks"].sum().unstack(fill_value=0)
    agg.columns = [f"clicks_{str(c).lower()}" for c in agg.columns]

    diversity = (
        merged.groupby(KEY_COLS)["activity_type"]
        .apply(lambda x: entropy(x.value_counts()))
        .rename("activity_diversity")
    )

    return agg.join(diversity, how="left").reset_index()


# ======================================================
# ASSESSMENTS
# ======================================================
def add_assessment_features(assess, assess_table, early_only=False):
    cols = ["id_assessment", "code_module", "code_presentation", "date"]
    df = assess.merge(assess_table[cols], on="id_assessment", how="left")
    df = _filter_early(df, early_only, day_col="date")

    if df.empty:
        return pd.DataFrame(columns=KEY_COLS + [
            "avg_score", "timed_score_mean", "num_assessments",
            "first_score", "last_score", "late_submissions",
            "avg_difficulty", "score_improvement", "missing_assessments"
        ])

    df["late_submission"] = (df["date_submitted"] > df["date"]).astype(int)

    difficulty = 1 - (
        df.groupby("id_assessment")["score"].mean() /
        (df.groupby("id_assessment")["score"].max() + 1e-9)
    )
    df["difficulty"] = df["id_assessment"].map(difficulty)

    df["timed_score"] = df["score"] * (1 - df["date"] / (df["date"].max() + 1))
    df = df.sort_values(KEY_COLS + ["date_submitted", "date"])

    summary = df.groupby(KEY_COLS).agg(
        avg_score=("score", "mean"),
        timed_score_mean=("timed_score", "mean"),
        num_assessments=("score", "count"),
        first_score=("score", "first"),
        last_score=("score", "last"),
        late_submissions=("late_submission", "sum"),
        avg_difficulty=("difficulty", "mean")
    ).reset_index()

    summary["score_improvement"] = summary["last_score"] - summary["first_score"]

    expected_table = assess_table.copy()
    expected_table = _filter_early(expected_table, early_only, day_col="date")
    expected = expected_table.groupby(["code_module", "code_presentation"])["id_assessment"] \
        .nunique().rename("expected_assessments").reset_index()
    summary = summary.merge(expected, on=["code_module", "code_presentation"], how="left")
    summary["expected_assessments"] = summary["expected_assessments"].fillna(0)
    summary["missing_assessments"] = summary["expected_assessments"] - summary["num_assessments"]
    summary = summary.drop(columns=["expected_assessments"])

    return summary


# ======================================================
# REGISTRATION
# ======================================================
def add_registration_features(registration):
    reg = registration.copy()
    reg["active_days"] = (reg["date_unregistration"] - reg["date_registration"]).fillna(0)
    reg["registered_late"] = (
        reg["date_registration"] >
        reg.groupby(["code_module", "code_presentation"])["date_registration"].transform("median")
    ).astype(int)

    return reg.groupby(KEY_COLS).agg(
        active_days=("active_days", "mean"),
        registered_late=("registered_late", "mean")
    ).reset_index()


# ======================================================
# LOGIN STREAKS
# ======================================================
def add_login_streaks(vle, early_only=False):
    df = _filter_early(vle.copy(), early_only, day_col="date")
    rows = []

    for keys, g in df.groupby(KEY_COLS):
        days = sorted(g["date"].dropna().unique())
        if len(days) == 0:
            rows.append([*keys, 0, 0])
            continue

        longest, current, max_gap = 1, 1, 0
        for i in range(1, len(days)):
            if days[i] == days[i - 1] + 1:
                current += 1
            else:
                longest = max(longest, current)
                max_gap = max(max_gap, days[i] - days[i - 1])
                current = 1

        rows.append([*keys, max(longest, current), max_gap])

    return pd.DataFrame(rows, columns=KEY_COLS + ["longest_streak", "max_gap"])


# ======================================================
# CLICK STATS
# ======================================================
def add_click_statistics(vle, early_only=False):
    df = _filter_early(vle.copy(), early_only, day_col="date")
    stats = df.groupby(KEY_COLS)["sum_click"].agg(
        click_mean="mean",
        click_std="std",
        click_min="min",
        click_max="max",
        click_skew=lambda x: _safe_skew(x),
        click_kurt=lambda x: _safe_kurt(x),
        click_iqr=lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    ).fillna(0)

    return stats.reset_index()


# ======================================================
# MAIN FEATURE BUILDER
# ======================================================
def build_full_features(student_info, registration, assessments, vle,
                        vle_meta, assess_table, courses, early_only=False):

    weekly = add_academic_weeks(vle, vle_meta, early_only=early_only)
    activity = add_activity_type_clicks(vle, vle_meta, early_only=early_only)
    assess = add_assessment_features(assessments, assess_table, early_only=early_only)
    reg = add_registration_features(registration)
    streaks = add_login_streaks(vle, early_only=early_only)
    click_stats = add_click_statistics(vle, early_only=early_only)

    vle_filtered = _filter_early(vle.copy(), early_only, day_col="date")
    total_clicks = (
        vle_filtered.groupby(KEY_COLS)["sum_click"]
        .sum()
        .reset_index(name="total_clicks")
    )

    df = student_info[
        KEY_COLS + [
            "age_band", "highest_education", "gender",
            "disability", "region", "final_result"
        ]
    ].copy()

    df = df.merge(weekly, on=KEY_COLS, how="left")
    df = df.merge(activity, on=KEY_COLS, how="left")
    df = df.merge(total_clicks, on=KEY_COLS, how="left")
    df = df.merge(assess, on=KEY_COLS, how="left")
    df = df.merge(reg, on=KEY_COLS, how="left")
    df = df.merge(streaks, on=KEY_COLS, how="left")
    df = df.merge(click_stats, on=KEY_COLS, how="left")

    df = df.merge(
        courses[["code_module", "code_presentation", "module_presentation_length"]],
        on=["code_module", "code_presentation"],
        how="left"
    )

    df["clicks_per_week"] = df["total_clicks"] / (df["module_presentation_length"] + 1)
    df = df.fillna(0)

    # Encode categoricals
    df["age_band"] = df["age_band"].map({"0-35": 0, "35-55": 1, "55<=": 2}).fillna(0)
    df["highest_education"] = df["highest_education"].map({
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }).fillna(0)
    df["gender"] = (df["gender"] == "M").astype(int)
    df["disability"] = (df["disability"] == "Y").astype(int)
    df["region_freq"] = np.log1p(df["region"].map(df["region"].value_counts()))

    # Labels
    df["final_label"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    # Treat withdrawal as at-risk for early intervention scenarios.
    df["at_risk"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    # ===================== EARLY =====================
    if early_only:
        week_cols = [
            c for c in df.columns
            if c.startswith("week_") and int(c.split("_")[1]) <= 3
        ]
        if len(week_cols) == 0:
            df["early_click_trend"] = 0.0
            df["early_click_volatility"] = 0.0
        else:
            df["early_click_trend"] = df[week_cols].diff(axis=1).mean(axis=1).fillna(0)
            df["early_click_volatility"] = df[week_cols].std(axis=1).fillna(0)

        features = week_cols + [
            "early_click_trend", "early_click_volatility",
            "avg_score", "timed_score_mean",
            "missing_assessments", "late_submissions",
            "activity_diversity", "longest_streak", "max_gap",
            "age_band", "highest_education", "gender",
            "disability", "region_freq"
        ]
        for c in features:
            if c not in df.columns:
                df[c] = 0

        X = (
            df[features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        return X, df["at_risk"]

    # ===================== FINAL =====================
    final_features = (
        [c for c in df.columns if c.startswith("week_")] +
        [c for c in df.columns if c.startswith("clicks_")] +
        [
            "total_clicks", "clicks_per_week",
            "avg_score", "timed_score_mean",
            "num_assessments", "missing_assessments",
            "first_score", "last_score", "score_improvement",
            "late_submissions", "active_days", "registered_late",
            "activity_diversity", "longest_streak", "max_gap",
            "avg_difficulty",
            "click_mean", "click_std", "click_min", "click_max",
            "click_skew", "click_kurt", "click_iqr",
            "age_band", "highest_education", "gender",
            "disability", "region_freq"
        ]
    )
    for c in final_features:
        if c not in df.columns:
            df[c] = 0

    X = (
        df[final_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    assert X.select_dtypes(include=["object"]).empty, \
        "Non-numeric columns detected in final features!"

    return X, df["final_label"]
