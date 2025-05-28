import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # 🧪**Overview of Hypothesis Testing**
    ---
    A statistical method to evaluate whether data supports or contradicts a particular *Hypothesis*. In the context of the medical insurance dataset, we'll be exploring the relationships between various factors like age, bmi, smoker, medical charges etc.

    We'll be stating Null Hypothesis (H0) & Alternative Hypothesis (H1). Running different Hypothesis using T-Test, Chi-Square Test and Power Analysis. Through this, we can gain valuable insights like factors that influence healthcare costs and develop evidence-based policies and strategies.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src=".assets\\null vs alt.png", width=750, height=450).center()
    return


@app.cell
def _(mo):
    # importing required libraries
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import statsmodels.api as sm
    import warnings

    # configuration
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.13, rc={"lines.linewidth": 3})


    # importing dataset
    df = pd.read_csv("insurance.csv", index_col="Unnamed: 0")
    df.head()

    mo.md("""
    Here is the Overview of *Medical Insurance Dataset 🏥*:

    {data}
    """).batch(data=mo.ui.table(df, selection="multi-cell"))
    return


if __name__ == "__main__":
    app.run()
