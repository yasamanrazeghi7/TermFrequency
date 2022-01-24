import random

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import logistic
from scipy.stats.mstats import mquantiles
import scipy.stats
import seaborn as sns
from typing import List

TOP_FREQ = 200
FREQ_NUM = 1
WORD_FLAG = False
FREQUENCY_DATA_X_COLUMN = 'testcase.data_point.frequency_data.x'
FREQUENCY_DATA_Y_COLUMN = 'testcase.data_point.frequency_data.y'
FREQUENCY_DATA_Z_COLUMN = 'testcase.data_point.frequency_data.z'
FREQUENCY_DATA_KEY_X_COLUMN = 'testcase.data_point.frequency_data.key'
FREQUENCY_DATA_KEY_XY_COLUMN = 'testcase.data_point.frequency_data.key_XY'
FREQUENCY_DATA_KEY_XZ_COLUMN = 'testcase.data_point.frequency_data.key_XZ'
FREQUENCY_DATA_KEY_YZ_COLUMN = 'testcase.data_point.frequency_data.key_YZ'
FREQUENCY_X_VALUE_COLUMN = 'testcase.data_point.frequency_data.frequency'
FREQUENCY_XY_VALUE_COLUMN = 'testcase.data_point.frequency_data.frequency_XY'
FREQUENCY_XZ_VALUE_COLUMN = 'testcase.data_point.frequency_data.frequency_XZ'
FREQUENCY_YZ_VALUE_COLUMN = 'testcase.data_point.frequency_data.frequency_YZ'
IS_CORRECT_COLUMN = 'is_correct'
NO_WORD_KEYS = ['mult', 'plus', 'concat', 'plushashtag', 'multhashtag', 'compareless', 'comparemore']


def log_reg_fit(X, y):
    X = X.reshape(-1, 1)
    model = LogisticRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_


class FrequencyCache:
    def __init__(self):
        self.frequency_dicts = {}

    def get_frequency_dict(self, word: str):
        file_name = "num_2_counting" if (word in NO_WORD_KEYS) else word
        if file_name in self.frequency_dicts:
            return self.frequency_dicts[file_name]
        num2_freq_file_path = f"./num2_counts/{file_name}.txt"
        frequency_dict = {}
        with open(num2_freq_file_path) as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                key, frequency = eval(line)
                frequency_dict[key] = frequency
                if (index+1) % 10_000_000 == 0:
                    print(f"I'm at index {index}, lines: {line}")
        self.frequency_dicts[file_name] = frequency_dict
        return self.frequency_dicts[file_name]


frequency_cache_singleton = FrequencyCache()


def find_pair_frequency(row, first_column: str, second_column: str, word):
    not_found = 0
    num_2_frequency = 0
    min_val = min(str(row[first_column]), str(row[second_column]))
    max_val = max(str(row[first_column]), str(row[second_column]))
    if word in NO_WORD_KEYS:
        tuple1 = (min_val, max_val)
        tuple2 = (max_val, min_val)
    else:
        tuple1 = (min_val, max_val, word)
        tuple2 = (max_val, min_val, word)
    freq_dict = frequency_cache_singleton.get_frequency_dict(word)
    if tuple2 in freq_dict.keys() and not (tuple1 == tuple2):
            print(f"something is very wrong!!!!!!!!!!! tuple1: {tuple1} tuple2: {tuple2}")
    if tuple1 in freq_dict.keys():
        num_2_key = str(tuple1)
        num_2_frequency = freq_dict[tuple1]
    else:
        num_2_key = str(tuple1)
        not_found = 1
    return num_2_key, num_2_frequency, not_found


class PlotInfo:
    def __init__(self, word: str, shots: int, model_key: str, key_type: str):
        """

        :param word:
        :param shots:
        :param model_key:
        :param key_type: It can be 'x', 'xy', or 'xz'
        """
        key_value_column_map = {
            'x': (FREQUENCY_DATA_KEY_X_COLUMN, FREQUENCY_X_VALUE_COLUMN),
            'xy': (FREQUENCY_DATA_KEY_XY_COLUMN, FREQUENCY_XY_VALUE_COLUMN),
            'xz': (FREQUENCY_DATA_KEY_XZ_COLUMN, FREQUENCY_XZ_VALUE_COLUMN),
            'yz': (FREQUENCY_DATA_KEY_YZ_COLUMN, FREQUENCY_YZ_VALUE_COLUMN),
        }
        if key_type not in key_value_column_map:
            raise Exception(f"Wrong Key!, key must be one of {key_value_column_map.keys()}")
        self.word = word
        self.shots = shots
        self.model_key = model_key
        self.key = key_type
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
        self.data_file.replace(False, 0, inplace=True)
        self.key_column, self.frequency_value_column = key_value_column_map[key_type]
        if key_type != 'x':
            if key_type == 'xy':
                first_column = FREQUENCY_DATA_X_COLUMN
                second_column = FREQUENCY_DATA_Y_COLUMN
            elif key_type == 'xz':
                first_column = FREQUENCY_DATA_X_COLUMN
                second_column = FREQUENCY_DATA_Z_COLUMN
            elif key_type == 'yz':
                first_column = FREQUENCY_DATA_Y_COLUMN
                second_column = FREQUENCY_DATA_Z_COLUMN
            else:
                raise Exception("Should not be here!")
            total_not_found = 0
            for i, row in self.data_file.iterrows():
                new_key, new_frequency, not_found = find_pair_frequency(row=row,
                                                                               first_column=first_column,
                                                                               second_column=second_column,
                                                                               word=self.word)
                self.data_file.at[i, self.key_column] = new_key
                self.data_file.at[i, self.frequency_value_column] = new_frequency if new_frequency > 0 else random.random()
                total_not_found += not_found
            if total_not_found > 0:
                print(f"Total Not Found for key {key_type} is {total_not_found}")
        self.aggregated_data_by_key = self.data_file.groupby(self.key_column)[
            [IS_CORRECT_COLUMN, self.frequency_value_column]].mean()

    def calculate_spearman(self):
        spearman_correlation = self.aggregated_data_by_key.corr(method='spearman')
        self.spearman_corr = spearman_correlation.loc[IS_CORRECT_COLUMN, self.frequency_value_column]
        return self

    def quantile_accuracies_plot(self, q_num: int = 10):
        frequencies = self.aggregated_data_by_key[self.frequency_value_column].to_numpy()
        is_correct = self.aggregated_data_by_key[IS_CORRECT_COLUMN].to_numpy()
        # Quantiles computes the bin edges
        quantiles = np.quantile(frequencies, q=np.linspace(0, 1, num=q_num + 1))
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
            h[bin_id - 1] = se * scipy.stats.t.ppf((1 + 0.95) / 2., n - 1)
        # this is for finding the mid point for error bars
        mid_quantiles = np.sqrt(quantiles[:-1] * quantiles[1:])
        self.quantile_bins['quantiles'] = quantiles
        self.quantile_bins['accuracies'] = accuracies
        self.quantile_bins['h'] = h
        self.quantile_bins['widths'] = widths
        self.quantile_bins['mid_quantiles'] = mid_quantiles
        self.similarity = accuracies[-1] - accuracies[0]
        return self

    def calculate_logistic_regression(self):
        freq_all = self.data_file[self.frequency_value_column].to_numpy()
        accuracy_all = self.data_file[IS_CORRECT_COLUMN].to_numpy()
        frequencies_log10 = np.ma.log10(freq_all)
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
        scatter_params['x'] = self.frequency_value_column
        scatter_params['y'] = IS_CORRECT_COLUMN
        return scatter_params

    def get_mode(self):
        word_to_mode_map = {
            'mult': 'Arithmetics-Multiplication',
            'plus': 'Arithmetics-Adding',
            'concat': 'Numbers-Concatination',
            'mode10hashtag': 'Numbers-Mode10#',
            'plushashtag': 'Numbers-Add#',
            'multhashtag': 'Numbers-Mult#',
            'compareless': 'Numbers-lower',
            'comparemore': 'Numbers-higher',
        }

        if not WORD_FLAG and self.word in word_to_mode_map:
            mode = word_to_mode_map[self.word]
        else:
            mode = f'Time-Coversion-From-{self.word}'
        return mode

    def get_model(self):
        model_name_map = {
            'gptjsmall': 'GPT-NEO-1.3B',
            'gptjlarge': 'GPT-NEO-2.7B',
            'gptj': 'GPT-J-6B'
        }
        return model_name_map[self.model_key]


def save_freq_acc_plot_and_get_info(word: str, shots: int, model: str, key: str, show_plot: bool = False, quantile_number: int = 10):
    # Creat PlotInfo
    plot_info = PlotInfo(word, shots, model, key_type=key)
    plot_info.calculate_spearman()
    plot_info.quantile_accuracies_plot(q_num=quantile_number)
    plot_info.calculate_logistic_regression()
    plot_info.calculate_accuracy_all()

    # Plot Configuration
    plt.xscale('log')

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
    plt.errorbar(mid_quantiles, accuracies, yerr=h, fmt="|", color="r")
    plt.plot(t, p_t, lw=1, ls="--", color="k", label="")

    # Scatter Chart
    scatter_params = plot_info.get_scatter_params()
    x = scatter_params['x']
    y = scatter_params['y']
    data_params = scatter_params['data']
    sns.regplot(data=data_params, x=x, y=y, scatter=True, fit_reg=False, scatter_kws={"color": "#18A558", "s": 5},
                x_jitter=0.01, y_jitter=0.01)

    # Plot Configuration (final)
    plt.xlabel('Frequency')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.60 )
    # plt.subplots_adjust(right=0.90 )
    plt.ylim([0, 1.05])

    shots_str = "0" + str(shots) if shots < 10 else str(shots)
    plt.savefig(f'./figures3/key_{key}_{plot_info.get_mode()}_{shots_str}shots_{plot_info.get_model()}.pdf', format='pdf',
                dpi=500)
    if show_plot:
        plt.show()
    return plot_info


def save_logistic_regression_lines_plot_for_shots(word: str, model: str, shots: List[int], key: str, show_plot: bool = False):
    plt.xscale('log')
    colors = ['#dda15e', '#e0aaff', '#06d6a0', '#073b4c', '#ef476f']
    line_style = [':', '-.', '--', '--', '-']
    mode = ""
    model_name = ""
    for i, shot in enumerate(shots):
        # Creat PlotInfo
        plot_info = PlotInfo(word, shot, model, key_type= key)
        mode = plot_info.get_mode()
        model_name = plot_info.get_model()
        # plot_info.calculate_spearman()
        # plot_info.quantile_accuracies_plot(q_num=10)
        plot_info.calculate_logistic_regression()
        # plot_info.calculate_accuracy_all()

        t = plot_info.logistic_regression_t
        p_t = plot_info.logistic_regression_pt
        plt.plot(t, p_t, lw=1, ls=line_style[i % len(line_style)], color=colors[i % len(colors)],
                 label=f"shots = {shot}")
    plt.legend(loc='center left', bbox_to_anchor=(0.70, 0.5))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Accuracy')
    plt.savefig(f'./figures4/key_{key}_mode_{mode}_shots_{model_name}.pdf', format='pdf', dpi=500)
    if show_plot:
        plt.show()


def save_logistic_regression_lines_plot_for_models(word: str, models: List[str], shot: int, key: str, show_plot: bool = False):
    plt.xscale('log')
    colors = ['#dda15e', '#e0aaff', '#06d6a0', '#073b4c', '#ef476f']
    line_style = [':', '-.', '--', '--', '-']
    mode = ""
    for i, model in enumerate(models):
        # Creat PlotInfo
        plot_info = PlotInfo(word, shot, model, key_type=key)
        mode = plot_info.get_mode()
        # plot_info.calculate_spearman()
        # plot_info.quantile_accuracies_plot(q_num=10)
        plot_info.calculate_logistic_regression()
        # plot_info.calculate_accuracy_all()

        t = plot_info.logistic_regression_t
        p_t = plot_info.logistic_regression_pt
        plt.plot(t, p_t, lw=1, ls=line_style[i % len(line_style)], color=colors[i % len(colors)],
                 label=f"LM = {plot_info.get_model()}")
    plt.legend(loc='center left', bbox_to_anchor=(0.70, 0.5))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Accuracy')
    plt.savefig(f'./figures4/key_{key}_mode_{mode}_shots_{shot}_all_models.pdf', format='pdf', dpi=500)
    if show_plot:
        plt.show()
