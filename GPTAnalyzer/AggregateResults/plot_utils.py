from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import logistic
from scipy.stats.mstats import mquantiles
import scipy.stats
import seaborn as sns

TOP_FREQ = 200
FREQ_NUM = 1
WORD_FLAG = False
FREQUENCY_DATA_KEY_COLUMN = 'testcase.data_point.frequency_data.key'
FREQUENCY_VALUE_COLUMN = 'testcase.data_point.frequency_data.frequency'
IS_CORRECT_COLUMN = 'is_correct'


def log_reg_fit(X, y):
    X = X.reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_


class PlotInfo:
    def __init__(self, word: str, shots: int, model_key: str):
        self.word = word
        self.shots = shots
        self.model_key = model_key
        self.spearman_corr = 0.0
        self.similarity = 0.0
        self.coef = 0.0
        self.intercept = 0.0
        self.intercept_50 = 0.0
        self.accuracy_all = 0.0
        self.quantile_bins = {
            'quantiles': np.array([]),
            'accuracies': np.array([]),
            'h': np.array([]),
            'widths': None,
            'mid_quantiles': None
        }
        self.logistic_regression_t = None
        self.logistic_regression_pt = None

        if self.word in ['mult', 'plus', 'concat', 'plushashtag', 'multhashtag']:
            file_name = f'./results2/results/num{FREQ_NUM}_{self.word}_1to50_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
        elif self.word in ['compareless', 'comparemore']:
            file_name = f'./results2/results/num{FREQ_NUM}_{self.word}_1to100_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
        else:
            file_name = f'./results2/results/num{FREQ_NUM}_{self.word}_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
        self.data_file = pd.read_csv(file_name)
        self.data_file.replace(True, 1, inplace=True)
        self.data_file.replace(False,0, inplace=True)
        self.aggregated_data_by_key = self.data_file.groupby(FREQUENCY_DATA_KEY_COLUMN)[IS_CORRECT_COLUMN, FREQUENCY_VALUE_COLUMN].mean()

    def calculate_spearman(self):
        spearman_correlation = self.aggregated_data_by_key.corr(method='spearman')
        self.spearman_corr = spearman_correlation.loc[IS_CORRECT_COLUMN, FREQUENCY_VALUE_COLUMN]
        return self

    def quantile_accuracies_plot(self, q_num: int = 10):
        frequencies = self.aggregated_data_by_key[FREQUENCY_VALUE_COLUMN].to_numpy()
        is_correct = self.aggregated_data_by_key[IS_CORRECT_COLUMN].to_numpy()
        # Quantiles computes the bin edges
        quantiles = np.quantile(frequencies, q=np.linspace(0, 1, num=q_num+1))
        quantiles[-1] += 1  # Edge case, we don't want biggest freq to be in its own bin
        widths = quantiles[1:] - quantiles[:-1]

        # Digitize assigns each frequency to it's bin
        frequency_bins = np.digitize(frequencies, bins=quantiles)
        # Compute the mean and error bars of is_correct in each bin
        accuracies = np.zeros(shape=(quantiles.shape[0] - 1,))  # -1 since there is 1 less bin than endpoints
        h = np.zeros(shape=(quantiles.shape[0] - 1,))
        for bin_id in range(frequency_bins.min(), frequency_bins.max() + 1):
            data_to_bin = is_correct[frequency_bins == bin_id]
            accuracies[bin_id - 1] = np.mean(data_to_bin)
            n = len(data_to_bin)
            se = scipy.stats.sem(data_to_bin)
            h[bin_id - 1] = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
        #this is for finding the mid point for error bars
        mid_quantiles = np.sqrt(quantiles[:-1]*quantiles[1:])
        self.quantile_bins['quantiles'] = quantiles
        self.quantile_bins['accuracies'] = accuracies
        self.quantile_bins['h'] = h
        self.quantile_bins['widths'] = widths
        self.quantile_bins['mid_quantiles'] = mid_quantiles
        self.similarity = accuracies[-1] - accuracies[0]
        return self

    def calculate_logistic_regression(self):
        freq_all = self.data_file[FREQUENCY_VALUE_COLUMN].to_numpy()
        accuracy_all = self.data_file[IS_CORRECT_COLUMN].to_numpy()
        frequencies_log10 = np.log10(freq_all)
        coef, intercept = log_reg_fit(frequencies_log10, accuracy_all)
        self.coef = coef
        self.intercept = intercept
        t = np.linspace(frequencies_log10.min(), frequencies_log10.max(), 10000)[:, None]
        p_t = logistic(t.T, -coef, -intercept)
        t = np.power(10, t)
        self.logistic_regression_t = t
        self.logistic_regression_pt = p_t
        return self

    def calculate_accuracy_all(self):
        self.accuracy_all = self.data_file[IS_CORRECT_COLUMN].to_numpy().mean()
        return self

    def get_scatter_params(self):
        scatter_params = {}
        scatter_params['data'] = self.aggregated_data_by_key
        scatter_params['x'] = FREQUENCY_VALUE_COLUMN
        scatter_params['y'] = IS_CORRECT_COLUMN
        return scatter_params

    def get_mode(self):
        word_to_mode_map = {
            'mult': 'Arithmetics-Multiplication',
            'plus': 'Arithmetics-Adding',
            'concat': 'Numbers-Concatination',
            'mode10hashtag' : 'Numbers-Mode10#',
            'plushashtag' : 'Numbers-Add#',
            'multhashtag' : 'Numbers-Mult#',
            'compareless' : 'Numbers-lower',
            'comparemore' : 'Numbers-higher',
        }

        if not WORD_FLAG and self.word in word_to_mode_map:
            mode = word_to_mode_map[self.word]
        else:
            mode = f'Time-Coversion-From-{self.word}'
        return mode

    def get_model(self):
        model_name_map = {
            'gptjsmall' : 'GPT-NEO-1.3B',
            'gptjlarge' : 'GPT-NEO-2.7B',
            'gptj' : 'GPT-J-6B'
        }
        return model_name_map[self.model_key]


def save_plot_and_get_info(word: str, shots: int, model: str, show_plot: bool = False):
    # Creat PlotInfo
    plot_info = PlotInfo(word, shots, model)
    plot_info.calculate_spearman()
    plot_info.quantile_accuracies_plot(q_num=10)
    plot_info.calculate_logistic_regression()
    plot_info.calculate_accuracy_all()

    # Plot Configuration
    plt.xscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.60 )
    # plt.subplots_adjust(right=0.90 )
    plt.ylim([0, 1.05])

    # Quantiles
    quantile_bins = plot_info.quantile_bins
    quantiles = quantile_bins['quantiles']
    accuracies = quantile_bins['accuracies']
    widths = quantile_bins['widths']
    h = quantile_bins['h']
    mid_quantiles = quantile_bins['mid_quantiles']
    plt.bar(quantiles[:-1], height=accuracies, width=widths, align='edge', alpha=0.3, edgecolor='#182e52')

    # Logistic Regression Line
    t = plot_info.logistic_regression_t
    p_t = plot_info.logistic_regression_pt
    plt.errorbar(mid_quantiles, accuracies, yerr=h ,fmt="|", color="r" )
    plt.plot(t, p_t, lw=1, ls="--", color="k", label="")

    # Scatter Chart
    scatter_params = plot_info.get_scatter_params()
    x = scatter_params['x']
    y = scatter_params['y']
    data_params = scatter_params['data']
    sns.regplot(data=data_params, x=x, y=y, scatter=True, fit_reg=False, scatter_kws={"color": "#18A558", "s":5}, x_jitter=0.01, y_jitter=0.01)

    shots_str = "0"+str(shots) if shots < 10 else str(shots)
    plt.savefig(f'./figures3/{plot_info.get_mode()}_{shots_str}shots_{plot_info.get_model()}.pdf', format='pdf', dpi=500)
    if show_plot:
        plt.show()
    return plot_info
