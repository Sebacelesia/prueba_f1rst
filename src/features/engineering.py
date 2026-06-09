import pandas as pd

COLS_TO_DROP = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeID']


def drop_useless(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sat_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement']
    df['satisfaction_index'] = df[sat_cols].mean(axis=1)

    df['company_tenure_ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)

    df['promotion_stagnation'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    avg_income_by_level = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')
    df['income_vs_level'] = (df['MonthlyIncome'] - avg_income_by_level) / avg_income_by_level

    df['manager_loyalty_ratio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)

    df['experience_density'] = df['TotalWorkingYears'] / (df['Age'] - 17)

    df['overwork_pressure'] = df['hrs'] / (df['WorkLifeBalance'] + 0.5)

    anchor_cols = ['YearsAtCompany', 'TotalWorkingYears', 'YearsWithCurrManager']
    df['anchoring_score'] = df[anchor_cols].mean(axis=1)

    return df
