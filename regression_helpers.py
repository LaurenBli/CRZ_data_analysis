import numpy as np
import pandas as pd


def extract_terms(model, terms, model_name):
    rows = []

    for term in terms:
        rows.append({
            "model": model_name,
            "term": term,
            "coef": model.params.get(term, np.nan),
            "std_err": model.bse.get(term, np.nan),
            "p_value": model.pvalues.get(term, np.nan),
            "ci_low": model.params.get(term, np.nan) - 1.96 * model.bse.get(term, np.nan),
            "ci_high": model.params.get(term, np.nan) + 1.96 * model.bse.get(term, np.nan),
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
        })

    return pd.DataFrame(rows)


def add_percent_effects(df):
    df = df.copy()
    df["approx_percent_effect"] = 100 * df["coef"]
    df["exact_percent_effect"] = 100 * (np.exp(df["coef"]) - 1)
    return df