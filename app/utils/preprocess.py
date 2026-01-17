import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis

BASE_PATH = "data/oulad/"


# ======================================================
# LOAD ALL OULAD TABLES
# ======================================================
def load_oulad():
    student_info = pd.read_csv(BASE_PATH + "studentInfo.csv")
    registration = pd.read_csv(BASE_PATH + "studentRegistration.csv")
    assessments = pd.read_csv(BASE_PATH + "studentAssessment.csv")
    vle = pd.read_csv(BASE_PATH + "studentVle.csv")
    vle_meta = pd.read_csv(BASE_PATH + "vle.csv")
    assessments_table = pd.read_csv(BASE_PATH + "assessments.csv")
    return student_info, registration, assessments, vle, vle_meta, assessments_table


# ======================================================
# TRUE ACADEMIC WEEK MAPPING (fixes date//7 error)
# ======================================================
def add_academic_weeks(vle, vle_meta):
    meta = vle_meta[['id_site', 'week_from', 'week_to']].copy()
    vle = vle.merge(meta, on="id_site", how="left")

    # use week_from as actual week number
    vle['week'] = vle['week_from'].fillna(0).astype(int)
    weekly = vle.groupby(['id_student', 'week'])['sum_click'].sum().unstack(fill_value=0)
    weekly.columns = [f"week_{int(c)}_clicks" for c in weekly.columns]

    return weekly.reset_index()


# ======================================================
# ACTIVITY-TYPE CLICKS WITH RARITY WEIGHTING
# ======================================================
def add_activity_type_clicks(vle, vle_meta):
    merged = vle.merge(vle_meta[['id_site', 'activity_type']], on="id_site", how="left")

    # rarity weight = log(total / count_per_type)
    type_counts = merged['activity_type'].value_counts()
    rarity = np.log(1 + (len(merged) / (type_counts + 1)))
    merged['rarity_weight'] = merged['activity_type'].map(rarity)

    merged['weighted_clicks'] = merged['sum_click'] * merged['rarity_weight']

    agg = merged.groupby(['id_student', 'activity_type'])['weighted_clicks'].sum().unstack(fill_value=0)
    agg.columns = [f"clicks_{str(c).lower()}" for c in agg.columns]

    # activity diversity (entropy)
    diversity = merged.groupby('id_student')['activity_type'].apply(lambda x: entropy(x.value_counts()))
    agg['activity_diversity'] = diversity

    return agg.reset_index()


# ======================================================
# ASSESSMENT FEATURES WITH DIFFICULTY
# ======================================================
def add_assessment_features(assess, assess_table):
    df = assess.merge(assess_table[['id_assessment', 'date', 'weight']], on="id_assessment", how="left")

    # difficulty = 1 - (avg_score / max_score)
    difficulty_map = df.groupby('id_assessment')['score'].mean()
    max_scores = df.groupby('id_assessment')['score'].max()
    diff = 1 - (difficulty_map / (max_scores + 1e-9))
    df['difficulty'] = df['id_assessment'].map(diff)

    df['late_submission'] = (df['date_submitted'] > df['date']).astype(int)

    summary = df.groupby('id_student').agg(
        avg_score=('score', 'mean'),
        num_assessments=('score', 'count'),
        first_score=('score', 'first'),
        last_score=('score', 'last'),
        late_submissions=('late_submission', 'sum'),
        avg_difficulty=('difficulty', 'mean')
    ).reset_index()

    summary['score_improvement'] = summary['last_score'] - summary['first_score']

    # missing assessment count
    expected = assess_table.groupby('id_assessment').size().shape[0]
    summary['missing_assessments'] = expected - summary['num_assessments']

    return summary


# ======================================================
# REGISTRATION BEHAVIOR FEATURES
# ======================================================
def add_registration_features(registration):
    reg = registration.copy()
    reg['active_days'] = reg['date_unregistration'] - reg['date_registration']
    reg['active_days'] = reg['active_days'].fillna(0)
    reg['registered_late'] = (reg['date_registration'] > reg['date_registration'].median()).astype(int)

    return reg.groupby("id_student").agg(
        active_days=('active_days', 'mean'),
        registered_late=('registered_late', 'mean')
    ).reset_index()


# ======================================================
# LOGIN STREAK & GAP
# ======================================================
def add_login_streaks(vle):
    streaks = []
    gaps = []

    for student_id, group in vle.groupby('id_student'):
        days = sorted(group['date'].unique())

        # streaks
        longest = 1
        current = 1

        # gaps
        max_gap = 0

        for i in range(1, len(days)):
            if days[i] == days[i-1] + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
                max_gap = max(max_gap, days[i] - days[i-1])

        longest = max(longest, current)

        streaks.append([student_id, longest, max_gap])

    df = pd.DataFrame(streaks, columns=["id_student", "longest_streak", "max_gap"])
    return df


# ======================================================
# CLICK STATISTICAL SIGNATURES
# ======================================================
def add_click_statistics(vle):
    stats = vle.groupby("id_student")['sum_click'].agg(
        click_mean='mean',
        click_std='std',
        click_min='min',
        click_max='max',
        click_skew=lambda x: skew(x),
        click_kurt=lambda x: kurtosis(x),
        click_iqr=lambda x: np.percentile(x, 75) - np.percentile(x, 25)
    ).fillna(0)

    return stats.reset_index()


# ======================================================
# MAIN FEATURE PIPELINE — LEVEL 3 FULL VERSION
# ======================================================
def build_full_features(student_info, registration, assessments, vle, vle_meta, assess_table, early_only=False):

    weekly = add_academic_weeks(vle, vle_meta)
    activity = add_activity_type_clicks(vle, vle_meta)
    assess = add_assessment_features(assessments, assess_table)
    reg = add_registration_features(registration)
    streaks = add_login_streaks(vle)
    click_stats = add_click_statistics(vle)

    total_clicks = vle.groupby("id_student")["sum_click"].sum().reset_index().rename(columns={"sum_click": "total_clicks"})

    df = student_info[['id_student', 'age_band', 'highest_education', 'gender', 'disability', 'region']].copy()

    df = df.merge(weekly, on="id_student", how="left")
    df = df.merge(activity, on="id_student", how="left")
    df = df.merge(total_clicks, on="id_student", how="left")
    df = df.merge(assess, on="id_student", how="left")
    df = df.merge(reg, on="id_student", how="left")
    df = df.merge(streaks, on="id_student", how="left")
    df = df.merge(click_stats, on="id_student", how="left")

    df = df.fillna(0)

    # demographics
    df['age_band'] = df['age_band'].map({'0-35': 0, '35-55': 1, '55>': 2, '55<=': 2}).fillna(0)
    df['highest_education'] = df['highest_education'].map({
        'No Formal quals': 0,
        'Lower Than A Level': 1,
        'A Level or Equivalent': 2,
        'HE Qualification': 3,
        'Post Graduate Qualification': 4
    }).fillna(0)
    df['gender'] = (df['gender'] == 'M').astype(int)
    df['disability'] = (df['disability'] == 'Y').astype(int)
    df['region_freq'] = np.log1p(df['region'].map(df['region'].value_counts()))

    # labels
    df['final_label'] = student_info['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    df['at_risk'] = student_info['final_result'].apply(lambda x: 1 if x in ['Fail', 'Withdrawn'] else 0)

    # ================================================================
    # EARLY MODEL FEATURES (weeks 0–3 only)
    # ================================================================
    if early_only:
        week_cols = [c for c in df.columns if c.startswith("week_") and int(c.split("_")[1]) <= 3]
        early_features = week_cols + [
            'avg_score', 'num_assessments', 'missing_assessments',
            'first_score', 'late_submissions',
            'age_band', 'highest_education', 'gender',
            'disability', 'region_freq',
            'activity_diversity', 'longest_streak', 'max_gap'
        ]
        return df[early_features], df['at_risk']

    # ================================================================
    # FINAL MODEL FEATURES
    # ================================================================
    final_features = (
        [c for c in df.columns if c.startswith("week_")] +
        [c for c in df.columns if c.startswith("clicks_")] +
        [
            'total_clicks', 'avg_score', 'num_assessments', 'missing_assessments',
            'first_score', 'last_score', 'score_improvement',
            'late_submissions', 'active_days', 'registered_late',
            'activity_diversity', 'longest_streak', 'max_gap',
            'avg_difficulty',
            'click_mean', 'click_std', 'click_min', 'click_max',
            'click_skew', 'click_kurt', 'click_iqr',
            'age_band', 'highest_education', 'gender',
            'disability', 'region_freq'
        ]
    )

    return df[final_features], df['final_label']
