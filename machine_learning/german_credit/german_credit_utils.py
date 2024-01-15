from ucimlrepo import fetch_ucirepo
import pandas as pd


def import_data():
    columns = ["checking_account_status",
               "duration",
               "credit_history",
               "purpose",
               "credit_amount",
               "saving_account",
               "present_employment_since",
               "installment_rate_percentage",
               "personal_status_and_sex",
               "other_debtors",
               "present_residence_since",
               "property",
               "age",
               "other installment plans",
               "housing",
               "number_of_bank_credits",
               "job",
               "number_of_people_to_provide_maintenance_for",
               "telephone",
               "foreign worker",
               "credit_risk"]

    statlog_german_credit_data = fetch_ucirepo(id=144)

    X = statlog_german_credit_data.data.features.copy()
    y = statlog_german_credit_data.data.targets.copy()

    X.rename(columns=dict(zip(X.columns, columns[:-1])), inplace=True)
    y.rename(columns=dict(zip(y.columns, columns[-1])), inplace=True)

    return X, y


def import_data_from_file():
    columns = ["checking_account_status",
               "duration",
               "credit_history",
               "purpose",
               "credit_amount",
               "saving_account",
               "present_employment_since",
               "installment_rate_percentage",
               "personal_status_and_sex",
               "other_debtors",
               "present_residence_since",
               "property",
               "age",
               "other installment plans",
               "housing",
               "number_of_bank_credits",
               "job",
               "number_of_people_to_provide_maintenance_for",
               "telephone",
               "foreign worker",
               "credit_risk"]

    with open("data/german.data") as f:
        lines = f.readlines()

    df = pd.DataFrame(columns=columns)
    for line in lines:
        df_row = list(filter(None, line.split(" ")))
        df_row.remove("\n")
        df_row = pd.DataFrame(dict(zip(columns, df_row)), columns=columns, index=[0])
        df = pd.concat([df, df_row], ignore_index=True)

    return df.loc[:, df.columns != "credit_risk"], df["credit_risk"]
