import marimo

__generated_with = "0.13.15"
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
    import altair as alt

    # configuration
    warnings.filterwarnings("ignore", category=FutureWarning)
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.13, rc={"lines.linewidth": 3})
    return alt, np, pd, stats


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
def _(alt, df, mo):
    # frequency fo each category
    print(df['bmi_category'].value_counts())

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
    return


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

    return normalweight_charges, obesity_charges, overweight_charges


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
    return (get_bootstrap_samples,)


@app.cell
def _(
    alt,
    get_bootstrap_samples,
    mo,
    normalweight_charges,
    obesity_charges,
    overweight_charges,
    pd,
):
    # get bootstrap samples for the charges based on bmi category
    normalweight_bootstrap_samples = get_bootstrap_samples(normalweight_charges,3000)

    overweight_bootstrap_samples = get_bootstrap_samples(overweight_charges,3000)

    obesity_bootstrap_samples = get_bootstrap_samples(obesity_charges,3000)


    # Create a combined DataFrame
    sdf = pd.DataFrame({
        'value': normalweight_bootstrap_samples.tolist() + 
                 overweight_bootstrap_samples.tolist() + 
                 obesity_bootstrap_samples.tolist(),
        'category': ['normal weight'] * len(normalweight_bootstrap_samples) + 
                    ['over weight'] * len(overweight_bootstrap_samples) + 
                    ['obesity'] * len(obesity_bootstrap_samples)
    })

    # Create the histogram
    cat_distribution_chart = alt.Chart(sdf).mark_bar(opacity=0.7, stroke='black').encode(
        alt.X('value:Q', bin=alt.Bin(maxbins=30), title='Value'),
        alt.Y('count()', stack=None, title='Frequency'),
        alt.Color('category:N', title='BMI Category')
    ).properties(
        width=600,
        height=400,
        title='Bootstrapped Samples for each BMI Category'
    ).configure_view(
        strokeWidth=0
    ).configure_title(
        fontSize=20
    )

    mo.ui.altair_chart(cat_distribution_chart, label="Distribution of Bootstrapped Samples").center()
    return (
        normalweight_bootstrap_samples,
        obesity_bootstrap_samples,
        overweight_bootstrap_samples,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    Why **bootstrapping**?? 

    The Central Limit Theorem states that the mean samples of data regardless of its distribution will follow normality if the sample size is greater. Since, our data satisifies CLT, bootstrapping can place. 

    > **From the Figure, we can say that the distribution is indeed normal on having samples 3000 for each `bmi_category`.**

    Let's see other transformations...
    """
    )
    return


@app.cell
def _(
    normalweight_bootstrap_samples,
    normalweight_charges,
    np,
    obesity_bootstrap_samples,
    obesity_charges,
    overweight_bootstrap_samples,
    overweight_charges,
    pd,
    stats,
):
    from sklearn.preprocessing import (PowerTransformer,
                                       QuantileTransformer,
                                       FunctionTransformer)

    # defining a function to generate default transformation to find the best fit for hypothesis
    def build_transformations(col_name:str)->dict:
        transformations = {
            "log":FunctionTransformer(np.log).fit_transform(col_name),
            "log10":FunctionTransformer(np.log10).fit_transform(col_name),
            "log2":FunctionTransformer(np.log2).fit_transform(col_name),
            "sqrt":FunctionTransformer(np.sqrt).fit_transform(col_name),
            "pow2":FunctionTransformer(lambda x:x**2,).fit_transform(col_name),
            "box-cox":PowerTransformer(method='box-cox',standardize=False).fit_transform(col_name.to_numpy().reshape(-1,1)).flatten(),
            "yeo-john":PowerTransformer(method='yeo-johnson',standardize=False).fit_transform(col_name.to_numpy().reshape(-1,1)).flatten(),
            "quantile":QuantileTransformer(n_quantiles=col_name.shape[0],output_distribution='normal').fit_transform(col_name.to_numpy().reshape(-1,1)).flatten()
        }

        return transformations


    # transformations for all bmi categories

    normal_weight_transformations = build_transformations(normalweight_charges)

    over_weight_transformations = build_transformations(overweight_charges)

    obesity_transformations = build_transformations(obesity_charges)

    # function to calculate shaprio results for each bmi category transformation
    get_shapiro_res = lambda transformation : list(map(lambda x: stats.shapiro(transformation[x])[0], transformation.keys()))

    transformation_data = pd.DataFrame({'transformation':list(normal_weight_transformations.keys()),
                           'normal_weight':get_shapiro_res(normal_weight_transformations),
                           'over_weight':get_shapiro_res(over_weight_transformations),
                           'obesity':get_shapiro_res(obesity_transformations)})

    # appending the bootstrap shapiro results
    transformation_data.loc[len(transformation_data)] = {'transformation':'bootstrap', 
                                'normal_weight':stats.shapiro(normalweight_bootstrap_samples)[0],
                                'over_weight':stats.shapiro(overweight_bootstrap_samples)[0],   
                                'obesity':stats.shapiro(obesity_bootstrap_samples)[0]}

    transformation_data
    return (transformation_data,)


@app.cell
def _(alt, mo, transformation_data):
    # Sample data (replace this with your actual DataFrame)
    # transformation_data = pd.read_csv("your_data.csv")

    # If you're using Jupyter or VSCode, enable Altair rendering
    alt.data_transformers.disable_max_rows()

    # Melt the data from wide to long format
    transformed = transformation_data.melt(id_vars="transformation", 
                            var_name="group",
                            value_name = "normality_score")

    # Title mapping (as in Seaborn)
    title_map = {
        "normal_weight": "normal weight charges",
        "over_weight": "over weight charges",
        "obesity": "obesity charges"
    }

    # Optional: Map group names to titles
    transformed["group"] = transformed["group"].map(title_map)



    # Create the Altair plot
    normality_scores_dev_chart = alt.Chart(transformed).mark_circle(size=200, opacity=0.7, stroke="black").encode(
        x=alt.X('normality_score:Q', title="normality_scores"),
        y=alt.Y('transformation:N', title="transformation"),
        color=alt.Color('group:N', legend=None),
        tooltip=['transformation', 'normality_score']
    ).properties(
        height=450,
        width=300
    ).facet(
        column=alt.Column('group:N', title=None, header=alt.Header(labelAngle=0))
    ).configure_axis(
        grid=True
    ).configure_view(
        stroke=None
    )

    mo.ui.altair_chart(normality_scores_dev_chart, label="Scores of Techniques around different Categories").center()
    return


@app.cell
def _(mo):
    mo.md(r""">**`quantile transformation` & `bootstrapping` are the optimal transformation which pleases to have normal distribution. Since QuantileTransformer targets the normal distribution by measuring in quantiles such that the outliers get squeezed. Parametric-Estimators like `box-cox` & `yeo-johnson` expects the input data to be normally distributed, hence we can't rely on that. Basic log-transformations achieve good shapiro scores since didn't outperform bootstrapping.**""")
    return


if __name__ == "__main__":
    app.run()
