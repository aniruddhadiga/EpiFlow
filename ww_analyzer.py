import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
import ordpy


class PermutationEntropyAnalyzer:
    def __init__(self, values, region='Region'):
        self.values = values
        self.region = region

    def compute_entropy(self, order_list=[3, 4], lag=1):
        N = len(self.values)
        result_frames = []

        for o in order_list:
            permu_entropy = {}
            weeks_shifts = np.arange(5, N, 1)

            for weeks in weeks_shifts:
                all_time_pe = []
                for start in range(0, N - weeks + 1):
                    segment = self.values[start:start + weeks]
                    pe = np.nan if weeks <= o else ordpy.permutation_entropy(segment, dx=o, dy=lag)
                    all_time_pe.append(pe)

                permu_entropy[weeks] = all_time_pe

            res_df = pd.DataFrame.from_dict(permu_entropy, orient='index').T
            res_df['ord'] = o
            result_frames.append(res_df)

        regres_df = pd.concat(result_frames).reset_index().rename(columns={'index': 'shift_index'})
        melted_df = regres_df.melt(id_vars=['shift_index', 'ord'], value_name='PE', var_name='win_len')
        melted_df['Predictability'] = 1 - melted_df['PE']
        return melted_df

    
class GetOptimalWindow:
    def __init__(self, df, value_col='Predictability', group_col='win_len'):
        self.df = df
        self.value_col = value_col
        self.group_col = group_col

    def run_tukey_test(self):
        self.df=self.df.groupby(['shift_index','win_len'],as_index=False).mean()

        self.df = self.df.dropna(subset=[self.value_col])
        tukey = pairwise_tukeyhsd(endog=self.df[self.value_col],
                                  groups=self.df[self.group_col],
                                  alpha=0.05)
        return tukey

    def format_tukey_results(self, tukey):
        tab = tukey.summary().as_csv().strip()
        sdf = pd.read_csv(StringIO(tab), sep=',', header=[1])
        sdf['reject'] = sdf['reject'].str.strip().map({'True': 1, 'False': 0})

        psdf = pd.DataFrame(index=range(5, 31), columns=range(5, 31))
        for i in psdf.index:
            for j in psdf.columns:
                try:
                    psdf.loc[i, j] = sdf[
                        (sdf.group1 == i) & (sdf.group2 == j)
                    ].reject.values[0]
                except:
                    psdf.loc[i, j] = np.nan
        return psdf

    def plot_heatmap(self, psdf, ax=None):
        colors = ('#00004c', '#ff0770')
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(psdf.astype(float), ax=ax, linewidths=0.05,
                    mask=psdf.isnull(), square=True, cmap=cmap,
                    cbar_kws={"shrink": 0.5})
        ax.collections[0].cmap.set_bad('0.85')

        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([0.25, 0.75])
        colorbar.set_ticklabels(['accept', 'reject'])
        return ax

    def get_optimal_window(self):
        tukey = self.run_tukey_test()
        psdf = self.format_tukey_results(tukey)
        win_len = psdf.loc[psdf.sum(axis=1) == 0].index.min()
        return win_len, psdf


from statsmodels.tsa.stattools import grangercausalitytests

class GrangerCausalityAnalyzer:
    def __init__(self, rdf, x1, x2):
        self.rdf = rdf.copy()
        self.rdf['time_value'] = pd.to_datetime(self.rdf['time_value'])
        self.x1 = x1
        self.x2 = x2
        self.gc_result = None

    def grangers_causation_matrix(self, data, variables, test='ssr_chi2test', verbose=False):
        maxlag = 2
        df = pd.DataFrame(columns=['X', 'Y'], index=[0, 1])
        k = 0
        for c in [self.x2, self.x1]:
            for r in [self.x2, self.x1]:
                if r != c:
                    test_result = grangercausalitytests(data[[r, c]].dropna(), maxlag=maxlag, verbose=False)
                    p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
                    if verbose:
                        print(f'Y = {r}, X = {c}, P Values = {p_values}')
                    min_p_value = np.min(p_values)
                    argmin_p_value = np.argmin(p_values)
                    if min_p_value > 0.05:
                        min_p_value = np.nan
                        argmin_p_value = np.nan
                    else:
                        argmin_p_value += 1
                    df.loc[k, 'X'] = c
                    df.loc[k, 'Y'] = r
                    df.loc[k, 'pval'] = min_p_value
                    df.loc[k, 'lag'] = argmin_p_value
                    k += 1
        return df

    def rolling_GC_test(self, win=18):
        gc = pd.DataFrame()
        dates = self.rdf[self.rdf.time_value > '2021-11-01'].time_value.unique()
        for dt in dates:
            try:
                windt = pd.to_datetime(dt) - pd.Timedelta(weeks=win)
                data = self.rdf[['time_value', self.x1, self.x2]].set_index('time_value')
                data = data.loc[windt:dt]
                temp = self.grangers_causation_matrix(data, variables=[self.x1, self.x2])
                temp['time_value'] = pd.to_datetime(dt)
                gc = pd.concat([gc, temp])
            except Exception as e:
                print(dt, e)
                continue
        self.gc_result = gc
        return gc

    def plot(self):
        if self.gc_result is None:
            raise ValueError("You need to run `rolling_GC_test()` before plotting.")

        gc = self.gc_result
        _, axr = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax = axr[0]

        for from_var, to_var, color in [(self.x2, self.x1, 'teal'), (self.x1, self.x2, 'orchid')]:
            xtemp = gc[gc.X == from_var].set_index('time_value')
            xtemp = xtemp.sort_index().asfreq('W-SAT')
            xtemp['lag'].plot(ax=ax, label=f'{from_var} --> {to_var}', drawstyle="steps", color=color, lw=3)

        ax.set_yticks([0.5, 1, 2, 2.25])
        ax.set_yticklabels(['', 'Lag 1', 'Lag 2', ''])
        ax.set_title('Rolling-Window Granger Causality', fontsize=15)
        ax.legend(fontsize=12, loc='lower right', ncol=2)

        ax2 = axr[1]
        ax3 = ax2.twinx()
        temp_r = self.rdf[self.rdf.time_value >= '2021-10-01'].dropna(subset=[self.x2])
        temp_r[['time_value', self.x1]].plot(x='time_value', y=self.x1, ax=ax2, color='orchid')
        temp_r[['time_value', self.x2]].plot(x='time_value', y=self.x2, ax=ax3, color='teal')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
