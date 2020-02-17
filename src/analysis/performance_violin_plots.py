import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
​
perf = pd.read_csv(Path("experiments/"
                        "kakaobrain_optimized_per_dataset_datasets_x_configs_evaluations/"
                        "perf_matrix.csv"), index_col=0)
validation = set(perf.index) - set(perf.columns)
perf = perf.drop(validation)
perf = perf.reset_index()
perf = perf.rename(columns={"index": "run_on"})
perf_long = pd.melt(perf, id_vars="run_on", var_name="config", value_name="mean_performance")

plt.figure(figsize=(40,8))
plt.minorticks_on()
ax = sns.violinplot(x="run_on", y="mean_performance", data=perf_long)
ax.tick_params(axis='x', which='minor', bottom=False)
ax.grid(b=True, which='major', axis="y")
ax.grid(b=True, which='minor', axis="y", alpha=0.3)
ax.set_ylim(top=1)
ax.get_figure().savefig("violin_plots.pdf")
