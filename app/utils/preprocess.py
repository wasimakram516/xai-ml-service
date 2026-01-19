import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis

BASE_PATH = "data/oulad/"


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


# ======================================================
# WEEKLY CLICKS
# ======================================================
def add_academic_weeks(vle, vle_meta):
    meta = vle_meta[['id_site', 'week_from']]
    vle = vle.merge(meta, on="id_site", how="left")
    vle['week'] = vle['week_from'].fillna(0).astype(int)

    weekly = vle.groupby(['id_student', 'week'])['sum_click'].sum().unstack(fill_value=0)
    weekly.columns = [f"week_{int(c)}_clicks" for c in weekly.columns]

    return weekly.reset_index()


# ======================================================
# ACTIVITY TYPES + ENTROPY
# ======================================================
def add_activity_type_clicks(vle, vle_meta):
    merged = vle.merge(vle_meta[['id_site', 'activity_type']], on="id_site", how="left")

    type_counts = merged['activity_type'].value_counts()
    rarity = np.log1p(len(merged) / (type_counts + 1))
    merged['weighted_clicks'] = merged['sum_click'] * merged['activity_type'].map(rarity)

    agg = merged.groupby(['id_student', 'activity_type'])['weighted_clicks'].sum().unstack(fill_value=0)
    agg.columns = [f"clicks_{c.lower()}" for c in agg.columns]

    agg['activity_diversity'] = merged.groupby('id_student')['activity_type'] \
        .apply(lambda x: entropy(x.value_counts()))

    return agg.reset_index()


# ======================================================
# ASSESSMENTS
# ======================================================
def add_assessment_features(assess, assess_table):
    df = assess.merge(
        assess_table[['id_assessment', 'date']],
        on="id_assessment",
        how="left"
    )

    df['late_submission'] = (df['date_submitted'] > df['date']).astype(int)

    difficulty = 1 - (
        df.groupby('id_assessment')['score'].mean() /
        (df.groupby('id_assessment')['score'].max() + 1e-9)
    )
    df['difficulty'] = df['id_assessment'].map(difficulty)

    df['timed_score'] = df['score'] * (1 - df['date'] / (df['date'].max() + 1))

    summary = df.groupby('id_student').agg(
        avg_score=('score', 'mean'),
        timed_score_mean=('timed_score', 'mean'),
        num_assessments=('score', 'count'),
        first_score=('score', 'first'),
        last_score=('score', 'last'),
        late_submissions=('late_submission', 'sum'),
        avg_difficulty=('difficulty', 'mean')
    ).reset_index()

    summary['score_improvement'] = summary['last_score'] - summary['first_score']

    expected = assess_table['id_assessment'].nunique()
    summary['missing_assessments'] = expected - summary['num_assessments']

    return summary


# ======================================================
# REGISTRATION
# ======================================================
def add_registration_features(registration):
    reg = registration.copy()
    reg['active_days'] = (reg['date_unregistration'] - reg['date_registration']).fillna(0)
    reg['registered_late'] = (reg['date_registration'] > reg['date_registration'].median()).astype(int)

    return reg.groupby('id_student').agg(
        active_days=('active_days', 'mean'),
        registered_late=('registered_late', 'mean')
    ).reset_index()


# ======================================================
# LOGIN STREAKS
# ======================================================
def add_login_streaks(vle):
    rows = []

    for sid, g in vle.groupby('id_student'):
        days = sorted(g['date'].unique())
        longest, current, max_gap = 1, 1, 0

        for i in range(1, len(days)):
            if days[i] == days[i-1] + 1:
                current += 1
            else:
                longest = max(longest, current)
                max_gap = max(max_gap, days[i] - days[i-1])
                current = 1

        rows.append([sid, max(longest, current), max_gap])

    return pd.DataFrame(rows, columns=['id_student', 'longest_streak', 'max_gap'])


# ======================================================
# CLICK STATS
# ======================================================
def add_click_statistics(vle):
    stats = vle.groupby('id_student')['sum_click'].agg(
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
# MAIN FEATURE BUILDER
# ======================================================
def build_full_features(student_info, registration, assessments, vle,
                        vle_meta, assess_table, courses, early_only=False):

    weekly = add_academic_weeks(vle, vle_meta)
    activity = add_activity_type_clicks(vle, vle_meta)
    assess = add_assessment_features(assessments, assess_table)
    reg = add_registration_features(registration)
    streaks = add_login_streaks(vle)
    click_stats = add_click_statistics(vle)

    total_clicks = (
        vle.groupby('id_student')['sum_click']
        .sum()
        .reset_index(name='total_clicks')
    )

    df = student_info[
        ['id_student', 'code_module', 'code_presentation',
         'age_band', 'highest_education',
         'gender', 'disability', 'region']
    ].copy()

    df = df.merge(weekly, on='id_student', how='left')
    df = df.merge(activity, on='id_student', how='left')
    df = df.merge(total_clicks, on='id_student', how='left')
    df = df.merge(assess, on='id_student', how='left')
    df = df.merge(reg, on='id_student', how='left')
    df = df.merge(streaks, on='id_student', how='left')
    df = df.merge(click_stats, on='id_student', how='left')

    df = df.merge(
        courses[['code_module', 'code_presentation', 'module_presentation_length']],
        on=['code_module', 'code_presentation'],
        how='left'
    )

    # Drop identifiers not used for learning
    df = df.drop(columns=['code_module', 'code_presentation'])

    df['clicks_per_week'] = df['total_clicks'] / (df['module_presentation_length'] + 1)
    df = df.fillna(0)

    # Encode categoricals
    df['age_band'] = df['age_band'].map({'0-35': 0, '35-55': 1, '55<=': 2}).fillna(0)
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

    # Labels
    df['final_label'] = student_info['final_result'].isin(
        ['Pass', 'Distinction']
    ).astype(int)
    df['at_risk'] = student_info['final_result'].isin(['Fail']).astype(int)

    # ===================== EARLY =====================
    if early_only:
        week_cols = [
            c for c in df.columns
            if c.startswith('week_') and int(c.split('_')[1]) <= 3
        ]

        df['early_click_trend'] = df[week_cols].diff(axis=1).mean(axis=1)
        df['early_click_volatility'] = df[week_cols].std(axis=1)

        features = week_cols + [
            'early_click_trend', 'early_click_volatility',
            'avg_score', 'timed_score_mean',
            'missing_assessments', 'late_submissions',
            'activity_diversity', 'longest_streak', 'max_gap',
            'age_band', 'highest_education', 'gender',
            'disability', 'region_freq'
        ]

        X = (
            df[features]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
        )

        return X, df['at_risk']

    # ===================== FINAL =====================
    final_features = (
        [c for c in df.columns if c.startswith('week_')] +
        [c for c in df.columns if c.startswith('clicks_')] +
        [
            'total_clicks', 'clicks_per_week',
            'avg_score', 'timed_score_mean',
            'num_assessments', 'missing_assessments',
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

    X = (
        df[final_features]
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

    # Final safety check (XGBoost 2.x)
    assert X.select_dtypes(include=['object']).empty, \
        "Non-numeric columns detected in final features!"

    return X, df['final_label']
