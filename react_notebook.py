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
    #### A statistical method to evaluate whether data supports or contradicts a particular *Hypothesis*. In the context of the medical insurance dataset, we'll be exploring the relationships between various factors like age, bmi, smoker, medical charges etc.

    #### We'll be stating Null Hypothesis (H0) & Alternative Hypothesis (H1). Running different Hypothesis using _T-Test, Chi-Square Test and Power Analysis_. Through this, we can gain valuable insights like factors that influence healthcare costs and develop evidence-based policies and strategies.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src=".assets\\null vs alt.png", width=750, height=450).center()
    return


@app.cell
def _():
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
    return np, pd, stats


@app.cell
def _(mo, pd):
    # importing dataset
    df = pd.read_csv("insurance.csv", index_col="Unnamed: 0")
    df.head()

    mo.md("""
    Here is the Overview of *Medical Insurance Dataset 🏥*:

    {data}
    """).batch(data=mo.ui.table(df, selection="multi-cell"))
    return (df,)


@app.cell
def _(mo, pd, stats):

    def shapiro_normality_test(x:pd.Series,repr:str):
        """
        Performs the Shapiro-Wilk Normality Test to check whether 
        the given data follows normal distribution of not.
        """
    
        statistic,p_value = stats.shapiro(x)
        if p_value > 0.05:
            print(f"{repr} : The data is normally distributed")
        else:
            print(f"{repr} : The data is not normally distributed")

        print(f"statistic={statistic.round(decimals=3)} \tp_value={p_value.round(decimals=3)}\n")

    mo.show_code()
    return (shapiro_normality_test,)


@app.cell
def _(mo):
    mo.md(text="""
    🚨**NOTE: *Shapiro-Wilk* Normality Test is sensitive to outliers. It can affect the p-value and would make deviations in the results. User other testing method like plotting qqplots etc.**
    """).callout(kind="warn").center()
    return


@app.cell
def _(df, mo):
    # frequency fo each category
    print(df['bmi_category'].value_counts())

    import altair as alt

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('bmi_category:N', title='BMI Category'),
        y=alt.Y('count():Q', title='Count'),
        color=alt.Color('bmi_category:N', scale=alt.Scale(scheme='pastel1')),
        stroke=alt.value('black')
    ).properties(
        title='BMI Category Counts',
        width=600,
        height=400
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        fontSize=20
    )

    mo.ui.altair_chart(chart, label="BMI Category Distribution").center()
    return (alt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""> **From the graph, Due to insufficient data points for the underweight category, we have excluded it from further analysis to ensure the reliability of our results.**""")
    return


@app.cell
def _(df, mo, shapiro_normality_test):
    # extracting medical charges for each bmi category
    normalweight_charges = df.query('bmi_category == "normal"')['charges']
    overweight_charges = df.query('bmi_category == "overweight"')['charges']
    obesity_charges = df.query('bmi_category == "obesity"')['charges']

    # ---------testing normality ---------
    with mo.redirect_stdout():
        # shapiro normality test for normal charges
        shapiro_normality_test(normalweight_charges,"normal weight")
        # shapiro normality test for overweight charges
        shapiro_normality_test(overweight_charges,"over weight")
        # shapiro normality test for obesity charges
        shapiro_normality_test(obesity_charges,"obesity") 

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(""">**The *Shapiro-Wilk* test indicates that the medical charges within each BMI category are not normally distributed. To address this, we'll working on basic as well as complex transformations of the data for each `bmi_category`.**""")
    return


@app.cell
def _(mo):
    mo.md("")
    mo.md(
        """  
    # 🛠️ **Data Transformation**
    ---
    """
    )
    return


@app.cell
def _(mo, np, pd, stats):
    def get_bootstrap_samples(data:pd.Series,n_resamples:int):
        """
        Generates bootstrap samples for given dataset

        Args:
            data:pd.Series
            n_resamples:int (total samples to generate)
        Returns:
            bootstrap_sample:numpy.ndarray
        """
        bootstrap_samples = stats.bootstrap(data=(data,),
                                            statistic=np.mean,
                                            confidence_level=0.95, # 5% significance level
                                            n_resamples=n_resamples,
                                            axis=0, 
                                            random_state=42).bootstrap_distribution
        return bootstrap_samples

    mo.show_code()
    return


@app.cell
def _(alt, mo, np, pd):
    np.random.seed(42)

    # Generating Data
    source = pd.DataFrame({
        'Trial A': np.random.normal(0, 0.8, 1000),
        'Trial B': np.random.normal(-2, 1, 1000),
        'Trial C': np.random.normal(3, 2, 1000)
    })

    h_chart = alt.Chart(source).transform_fold(
        ['Trial A', 'Trial B', 'Trial C'],
        as_=['Experiment', 'Measurement']
    ).mark_bar(
        opacity=0.3,
        binSpacing=0
    ).encode(
        alt.X('Measurement:Q').bin(maxbins=100),
        alt.Y('count()').stack(None),
        alt.Color('Experiment:N')
    )

    mo.ui.altair_chart(h_chart)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
