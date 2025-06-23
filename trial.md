---
title: Trial
marimo-version: 0.13.11
width: full
html_head_file: head.html
---

```python {.marimo}
import marimo as mo
```

# 🧪**Hypothesis Testing on Medical Insurance Dataset**
---
#### A statistical method to evaluate whether data supports or contradicts a particular *Hypothesis*. In the context of the medical insurance dataset, we'll be exploring the relationships between various factors like age, bmi, smoker, medical charges etc.

#### We'll be stating Null Hypothesis (H0) & Alternative Hypothesis (H1). Running different Hypothesis using _T-Test, Chi-Square Test and Power Analysis_. Through this, we can gain valuable insights like factors that influence healthcare costs and develop evidence-based policies and strategies.

```python {.marimo hide_code="true"}
mo.image(src=".assets\\null vs alt.png", width=750, height=450).center()
```

```python {.marimo}
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
```

```python {.marimo}
# importing dataset
df = pd.read_csv("insurance.csv", index_col="Unnamed: 0")
df.head()

mo.md("""
<br>
**Here is the Overview of Medical Insurance Dataset 🏥**:

{data}
""").batch(data=mo.ui.table(df, selection="multi-cell"))
```

```python {.marimo}

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
```

```python {.marimo}
mo.md(text="""
🚨**NOTE: *Shapiro-Wilk* Normality Test is sensitive to outliers. It can affect the p-value and would make deviations in the results. User other testing method like plotting qqplots etc.**
""").callout(kind="warn").center()
```

```python {.marimo}

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
```

>**From the graph, Due to insufficient data points for the underweight category, we have excluded it from further analysis to ensure the reliability of our results**
<br>

```python {.marimo}
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

```

>**The *Shapiro-Wilk* test indicates that the medical charges within each BMI category are not normally distributed. To address this, we'll working on basic as well as complex transformations of the data for each `bmi_category`.**
<!---->
<br>
## **🛠 Data Transformation: *Finding the optimal approach***
---

```python {.marimo}
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
```

```python {.marimo}
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
```

```python {.marimo}
mo.callout(mo.md(
    r"""
### **Why bootstrapping**?? 

The Central Limit Theorem states that the mean samples of data regardless of its distribution will follow normality if the sample size is greater. Since, our data satisifies CLT, bootstrapping can place. 

**From the Figure, we can say that the distribution is indeed normal on having samples 3000 for each `bmi_category`.**
"""
), kind="info")
```

```python {.marimo}
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
```

```python {.marimo}

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
```

`quantile transformation` & `bootstrapping` are the optimal transformation which pleases to have normal distribution. Since QuantileTransformer targets the normal distribution by measuring in quantiles such that the outliers get squeezed. Parametric-Estimators like `box-cox` & `yeo-johnson` expects the input data to be normally distributed, hence we can't rely on that. Basic log-transformations achieve good shapiro scores since didn't outperform bootstrapping.
<br>
<br>

```python {.marimo}
import sklearn.metrics as metric

normal_weight_transformations['bootstrap'] = normalweight_bootstrap_samples

over_weight_transformations['bootstrap'] = overweight_bootstrap_samples

obesity_transformations['bootstrap'] = obesity_bootstrap_samples

class GraphQQ:

    def __init__(self, sample_data:pd.Series):
        self.sample_quantiles = np.sort(sample_data)
        self.theoretical_quantiles = self.compute_theoretical_quantiles(self.sample_quantiles)


    @staticmethod
    def compute_theoretical_quantiles(sq):
        """
        getting the quantiles w.r.t. the size, std.dev, mean of the sample quantiles
        """
        n = len(sq)
        sigma = np.std(sq,ddof=1)
        mu = np.mean(sq)

        probs = (np.arange(1,n+1)-0.5)/n # rank base probabilities
        return stats.norm.ppf(probs) * sigma + mu


    def generate_qq_plot(self, category:str, c:str):
        """
        generate the required qq plot
        """
        temp_df = pd.DataFrame({
            'Theoretical Quantiles': self.theoretical_quantiles,
            'Sample Quantiles': self.sample_quantiles
        })

        min_q = min(self.sample_quantiles.min(), self.theoretical_quantiles.min())
        max_q = max(self.sample_quantiles.max(), self.theoretical_quantiles.max())


        scatter_plot = alt.Chart(temp_df).mark_point(size=80,color=c).encode(
        x=alt.X('Theoretical Quantiles', scale=alt.Scale(domain=(min_q, max_q))),
        y=alt.Y('Sample Quantiles', scale=alt.Scale(domain=(min_q, max_q)))
        ).properties(
            width=300,
            height=200,
            title=category
        )   
        # creating the reference line
        ref_line = alt.Chart(pd.DataFrame({
        'x': [min_q, max_q],
        'y': [min_q, max_q]
        })).mark_line(color='red').encode(
            x='x',
            y='y'
        )
        # joining the plots
        qq_plot = scatter_plot + ref_line
        return qq_plot

    @staticmethod
    def get_scores(sq,tq)->dict:
        """scores to observe linear relationship"""
        r2_score = metric.r2_score(y_true=tq,y_pred=sq)
        mse = metric.mean_squared_error(y_true=tq,y_pred=sq)
        mae = metric.mean_absolute_error(y_true=tq,y_pred=sq)
        rmse = np.sqrt(mse)
        return {"R2 SCORE":r2_score, "MEAN SQ. ERROR":mse, "MEAN ABS. ERROR":mae, "ROOT MSE":rmse}
```

```python {.marimo name="input_radio"}
# radio 
input_method = mo.ui.radio.from_series(transformation_data['transformation'],
                                       label="Transformation Method:",
                                       value="log10", inline=True)
```

```python {.marimo name="graph_stack"}
# graph objects for each category

normal_weight_graph_obj = GraphQQ(sample_data=normal_weight_transformations[input_method.value])
over_weight_graph_obj = GraphQQ(sample_data=over_weight_transformations[input_method.value])
obesity_graph_obj = GraphQQ(sample_data=obesity_transformations[input_method.value])

# graph stack
graph_stack = mo.hstack([
    normal_weight_graph_obj.generate_qq_plot(category="Normal Weight Samples",c="#1f77b4"),
    over_weight_graph_obj.generate_qq_plot(category="Over Weight Samples", c="dimgrey"),
    obesity_graph_obj.generate_qq_plot(category="Obesity Samples", c="#7b68ee"),
], align="center", justify="center")


```

```python {.marimo name="score_stack"}
# score stack

total_sq = np.concatenate([normal_weight_graph_obj.sample_quantiles,
                over_weight_graph_obj.sample_quantiles,
                obesity_graph_obj.sample_quantiles])

total_tq = np.concatenate([normal_weight_graph_obj.theoretical_quantiles,
                over_weight_graph_obj.theoretical_quantiles,
                obesity_graph_obj.theoretical_quantiles])

scores = normal_weight_graph_obj.get_scores(sq=total_sq,tq=total_tq) # any graph obj can be used

stats_lst = [mo.stat(label=name,value=scores[name],bordered=True,direction="increase") for name in scores]
scores_stack = mo.hstack(stats_lst, justify="center", gap=1.4, align="center")
```

```python {.marimo}
mo.vstack(items=[
    input_method,
    mo.md("<br>"),
    graph_stack,
    mo.md("<br>"),
    scores_stack
])
```

<br>
## **💨 Running Hypothesis**
---
#### Now, after getting the data transformed and finding the optimal transformation i.e. bootstrapping. We're good to perform hypothesis on the dataset. Each problem uses different test to evaluate the estimations based on our objective.
<br>
<!---->
### **HP,1 : Is there a statistically significant difference between the population mean medical charge and the sample mean charges based on BMI category?**

**Null Hypothesis (H0)**:
No statistically significant difference exists between the population and sample mean charges.

**Alternative Hypothesis (H1)**:
A statistically significant difference exists between the population and sample mean charges.


#### **One Sample T-Test**
***Assumptions***:<br>
1. **Normality**: Data is normally distributed<br>
2. **Independence**: Data points are independent of each other<br>
3. **Randomness**: Data is randomly sampled from the population<br>
4. **Representativeness**: Data is representative of the population<br>

```python {.marimo name="one_sample_t_test"}
# retrieving the population mean charges
pop_mean = df['charges'].mean()

# defining the function which performs T-Test
def one_sample_t_test(sample: pd.Series, pop_mean: float) -> tuple:
    """
    Performs a one-sample t-test to determine if there is a significant difference 
    between the population mean charges and the sample mean charges.

    Args:
        sample (pd.Series): A sample of data to compare to the population mean.
        pop_mean (float): The population mean to compare the sample to.
    """
    t_statistic, p_value = stats.ttest_1samp(sample, pop_mean)
    t_statistic = round(t_statistic, 3)
    p_value = round(p_value, 3)
    print(f"t_statistic={t_statistic}")
    print(f"p_value={p_value}")
    print(f"There is a {'no ' if p_value >= 0.05 else ''}significant difference "
          f"between the population mean charges and the sample mean charges.")

```

```python {.marimo}
mo.ui.dropdown(["aloo","pudi"], value="aloo", label="kya khayega?")
```