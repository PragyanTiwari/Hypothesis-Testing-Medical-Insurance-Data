## qqplot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats


df = pd.read_csv('insurance.csv')
normal_weight_charges = df.query("bmi_category == 'normal'")['charges']

# transformed data

transformations = {"normal":normal_weight_charges,
                   "log10":np.log10(normal_weight_charges),
                   "log2":np.log2(normal_weight_charges),
                   "log":np.log(normal_weight_charges),
                   "pow":np.power(normal_weight_charges,2),
                   "sqrt":np.sqrt(normal_weight_charges)}
data = pd.DataFrame(transformations)
fig, axes = plt.subplots(nrows=1, ncols=len(data.columns), figsize=(15, 5))

# Plot Q-Q plots for each column
for i, col in enumerate(data.columns):
    stats.probplot(data[col], dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q Plot for {col}")
    axes[i].set_xlabel("Theoretical Quantiles")
    axes[i].set_ylabel("Sample Quantiles")

plt.tight_layout()
plt.show()


stats.probplot(transformations['log10'],dist='norm',plot=plt,col)
stats.probplot(transformations['sqrt'],dist='norm',plot=plt)
plt.show()

sns.pr