import random

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import logistic
from scipy.stats.mstats import mquantiles
import scipy.stats
import seaborn as sns
import operator
from typing import Callable, Any, List
from functools import reduce

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

# Plot constants
ALPHA = 0.175
COLORS = ['#CE822C', '#C561FF', '#03A57A', '#169AC5', '#F26385']
LINESTYLES = ['--', '-.', '-', ':']


def convert_any(i: Any, is_timeunit: bool) -> List[int]:
    if isinstance(i, int):
        return [i]
    elif isinstance(i, str):
        if is_timeunit:
            return [int(x) for x in eval(i)[:-1]]
        return [int(x) for x in eval(i)]
    raise Exception("Not supported!")


def default_index_score(list_i: List[int], list_j: List[int]) -> float:
    scores = [0.01 if i == j else abs(i-j) for (i, j) in zip(list_i, list_j)]
    return reduce(operator.mul, scores)**2


def default_index_filter(list_i: List[int]):
    return not any((len(str(i)) < 2 or i % 10 == 0) for i in list_i)


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
                if (index + 1) % 10_000_000 == 0:
                    print(f"I'm at index {index}, lines: {line}")
        self.frequency_dicts[file_name] = frequency_dict
        return self.frequency_dicts[file_name]


frequency_cache_singleton = FrequencyCache()


def find_pair_frequency(row, first_column: str, second_column: str, word, my_frequency_cache: FrequencyCache):
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
    freq_dict = my_frequency_cache.get_frequency_dict(word)
    if tuple2 in freq_dict.keys() and not (tuple1 == tuple2):
        print(f"something is very wrong!!!!!!!!!!! tuple1: {tuple1} tuple2: {tuple2}")
    if tuple1 in freq_dict.keys():
        num_2_key = str(tuple1)
        num_2_frequency = freq_dict[tuple1]
    else:
        num_2_key = str(tuple1)
        not_found = 1
    return num_2_key, num_2_frequency, not_found


def extrap(x, y):
    # Returns a function for extrapolating a linear fit between
    # two points (in log-scaled x-axis).
    x = np.log(x)
    slope = (y[1] - y[0])/(x[1] - x[0])
    intercept = y[0] - slope * x[0]
    return lambda x: slope * np.log(x) + intercept


class PlotInfo:
    def __init__(self, word: str, shots: int, model_key: str, key_type: str, my_frequency_cache: FrequencyCache = None):
        """

        :param word:
        :param shots:
        :param model_key:
        :param key_type: It can be 'x', 'xy', or 'xz'
        """
        if my_frequency_cache is None:
            my_frequency_cache = frequency_cache_singleton
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
            file_name = f'./results5/results/num{FREQ_NUM}_{self.word}_1to50_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
        elif self.word in ['compareless', 'comparemore', 'comparemoreless']:
            file_name = f'./results5/results/num{FREQ_NUM}_{self.word}_1to100_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
        else:
            file_name = f'./results5/results/num{FREQ_NUM}_{self.word}_top{TOP_FREQ}_{self.model_key}_{self.shots}shots_5seeds_results.csv'
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
                                                                        word=self.word,
                                                                        my_frequency_cache=my_frequency_cache)
                self.data_file.at[i, self.key_column] = new_key
                self.data_file.at[
                    i, self.frequency_value_column] = new_frequency if new_frequency > 0 else random.random()
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
        
        # this is the dilated interpolation point
        t = np.linspace(0, 1, num=q_num)
        log_quantiles = np.log(quantiles)
        left_endpoints = log_quantiles[:-1]
        right_endpoints = log_quantiles[1:]
        log_dilated_quantiles = (
            (1 - t) * left_endpoints + t * right_endpoints
        )
        dilated_quantiles = np.exp(log_dilated_quantiles)
        
        # this is extrapolated to make interpolated curve's range cover 
        # all of the data
        extrap_quantiles = np.concatenate((quantiles[:1], mid_quantiles, quantiles[-1:]))
        left_acc = extrap(mid_quantiles[:2], accuracies[:2])(quantiles[:1])
        right_acc = extrap(mid_quantiles[-2:], accuracies[-2:])(quantiles[-1:])
        extrap_accuracies = np.concatenate((left_acc, accuracies, right_acc))
        extrap_h = np.concatenate((h[:1], h, h[-1:]))
        
        
        self.quantile_bins['quantiles'] = quantiles
        self.quantile_bins['accuracies'] = accuracies
        self.quantile_bins['h'] = h
        self.quantile_bins['widths'] = widths
        self.quantile_bins['mid_quantiles'] = mid_quantiles
        self.quantile_bins['dilated_quantiles'] = dilated_quantiles
        self.quantile_bins['extrap_quantiles'] = extrap_quantiles
        self.quantile_bins['extrap_accuracies'] = extrap_accuracies
        self.quantile_bins['extrap_h'] = extrap_h
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
            'gptjsmall': '1.3B',
            'gptjlarge': '2.7B',
            'gptj': '6B'
        }
        return model_name_map[self.model_key]


def save_freq_acc_plot_and_get_info(
    word: str, shots: int, model: str, key: str,
    show_plot: bool = False,
    quantile_number: int = 10,
    important_points = None,
    plot_log_reg: bool = True,
    plot_lin_interp: bool = True,
    plot_scatters: bool = True,
    plot_bar: bool = True,
    plot_err: bool = True,
):
    if important_points is None:
        important_points = []
        
    # Create PlotInfo
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
    
    if plot_bar:
        plt.bar(quantiles[:-1], height=accuracies, width=widths, align='edge', alpha=ALPHA, edgecolor='#182e52')
        plt.errorbar(mid_quantiles, accuracies, yerr=h, fmt="|", color="r")

    # Logistic Regression Line
    t = plot_info.logistic_regression_t
    p_t = plot_info.logistic_regression_pt

    if plot_log_reg:
    # plt.xlim([t[], 1.05])
        plt.plot(t, p_t, lw=1, ls="--", color="k", label="")
        
    if plot_lin_interp:
        plt.plot(quantile_bins['extrap_quantiles'],
                 quantile_bins['extrap_accuracies'],
                 lw=1, ls=LINESTYLES[0], color=COLORS[0], label="")
    if plot_err:
        plt.fill_between(
            quantile_bins['extrap_quantiles'],
            quantile_bins['extrap_accuracies'] - quantile_bins['extrap_h'],
            quantile_bins['extrap_accuracies'] + quantile_bins['extrap_h'],
            color=COLORS[0],
            alpha=ALPHA
        )

    # Scatter Chart
    scatter_params = plot_info.get_scatter_params()
    x = scatter_params['x']
    y = scatter_params['y']
    data_params = scatter_params['data']
    if plot_scatters:
        sns.regplot(data=data_params, x=x, y=y, scatter=True, fit_reg=False, 
                    scatter_kws={"color": COLORS[0], "s": 5},
                    x_jitter=0.01, y_jitter=0.01)

    if important_points:
        my_new_data_frame = {x: [], y: [], 'label': []}
        for index, row in data_params.iterrows():
            if index in important_points:
                my_new_data_frame[x].append(row[x])
                my_new_data_frame[y].append(row[y])
                label = " ".join(str(q) for q in convert_any(index, plot_info.word not in NO_WORD_KEYS))
                my_new_data_frame['label'].append(label)
        my_new_data_frame = pd.DataFrame(my_new_data_frame)
        p1 = sns.regplot(data=my_new_data_frame, x=x, y=y, scatter=True, fit_reg=False, scatter_kws={"color": "#000000", "s": 30},
                    x_jitter=0.0, y_jitter=0.0, marker="+")
        for line in range(0, my_new_data_frame.shape[0]):
            p1.text(my_new_data_frame.loc[line][x] + 0.05, my_new_data_frame.loc[line][y]+0.03,
                    my_new_data_frame.loc[line]['label'], horizontalalignment='left',
                    size='medium', color='black', weight='semibold')

    # Plot Configuration (final)
    plt.xlabel('Frequency')
    plt.ylabel('Avg. Accuracy')
    plt.tight_layout()
    # plt.subplots_adjust(top=0.60 )
    # plt.subplots_adjust(right=0.90 )
    plt.ylim([0, 1.05])
    shots_str = "0" + str(shots) if shots < 10 else str(shots)
    plt.savefig(f'./figures4/key_{key}_{plot_info.get_mode()}_{shots_str}shots_{plot_info.get_model()}.pdf',
                format='pdf',
                dpi=500)
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()
    return plot_info


def save_logistic_regression_lines_plot_for_shots(
        word: str, model: str, shots: List[int], key: str,
        show_plot: bool = False
):
    plt.xscale('log')
    mode = ""
    model_name = ""
    for i, shot in enumerate(shots):
        
        # Create PlotInfo
        plot_info = PlotInfo(word, shot, model, key_type=key)
        mode = plot_info.get_mode()
        model_name = plot_info.get_model()
        # plot_info.calculate_spearman()
        # plot_info.quantile_accuracies_plot(q_num=10)
        plot_info.calculate_logistic_regression()
        # plot_info.calculate_accuracy_all()

        t = plot_info.logistic_regression_t
        p_t = plot_info.logistic_regression_pt
        plt.plot(t, p_t, lw=1,
                 ls=LINESTYLES[i % len(LINESTYLES)],
                 color=COLORS[i % len(COLORS)],
                 label=f"shots = {shot}")
    plt.legend(loc='center left', bbox_to_anchor=(0.70, 0.5))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(f'./figures4/key_{key}_mode_{mode}_shots_{model_name}.pdf', format='pdf', dpi=500)
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()
        

def save_logistic_regression_lines_plot_for_models(
        word: str, models: List[str], shot: int, key: str,
        show_plot: bool = False
):
    plt.xscale('log')
    mode = ""
    for i, model in enumerate(models):
        
        # Create PlotInfo
        plot_info = PlotInfo(word, shot, model, key_type=key)
        mode = plot_info.get_mode()
        # plot_info.calculate_spearman()
        # plot_info.quantile_accuracies_plot(q_num=10)
        plot_info.calculate_logistic_regression()
        # plot_info.calculate_accuracy_all()

        t = plot_info.logistic_regression_t
        p_t = plot_info.logistic_regression_pt
        plt.plot(t, p_t, lw=1,
                 ls=LINESTYLES[i % len(LINESTYLES)],
                 color=COLORS[i % len(COLORS)],
                 label=f"LM = {plot_info.get_model()}")
    plt.legend(loc='center left', bbox_to_anchor=(0.70, 0.5))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(f'./figures4/key_{key}_mode_{mode}_shots_{shot}_all_models.pdf', format='pdf', dpi=500)
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()


def save_interpolation_lines_plot_for_models(
        word: str, models: List[str], shot: int, key: str,
        quantile_number: int = 10, show_plot: bool = False
):
    plt.xscale('log')
    mode = ""
    model_name = ""
    for i, model in enumerate(models):
        plot_info = PlotInfo(word, shot, model, key_type=key)
        model_name = plot_info.get_model()
        plot_info.quantile_accuracies_plot(q_num=quantile_number)

        quantile_bins = plot_info.quantile_bins
        quantiles = quantile_bins['extrap_quantiles']
        accuracies = quantile_bins['extrap_accuracies']
        h = quantile_bins['extrap_h']
        plt.errorbar(quantiles, accuracies, yerr=h, fmt="|", color=COLORS[i % len(COLORS)])
        plt.plot(quantiles, accuracies, lw=1, 
                 ls=LINESTYLES[j % len(LINESTYLES)],
                 color=COLORS[i % len(COLORS)],
                 label=f"LM = {plot_info.get_model()}")
    plt.legend(loc='center left', bbox_to_anchor=(0.6, 0.6))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(
        f'./figures4/key_{key}_mode_{mode}_shots_{shot}__all_models_interp.pdf',
        format='pdf',
        dpi=500,
        bbox_inches='tight'
    )
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()
    

def save_logistic_regression_lines_plot_for_models_shots(word: str, models: List[str], shots: List[int], key: str,
                                                         show_plot: bool = False):
    plt.xscale('log')
    mode = ""
    for j, shot in enumerate(shots):
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
            plt.plot(t, p_t, lw=1, ls=LINESTYLES[j % len(LINESTYLES)], color=COLORS[i % len(COLORS)],
                     label=f"k={shot}, LM{plot_info.get_model()}")
    plt.legend(loc='center left', bbox_to_anchor=(0.60, 0.6))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(f'./figures4/key_{key}_mode_{mode}_some_shots__all_models.pdf', format='pdf', dpi=500, bbox_inches='tight')
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()


def save_interpolation_lines_plot_for_models_shots(word: str, models: List[str], shots: List[int], key: str, quantile_number: int = 10, show_plot: bool = False):
    plt.xscale('log')#169AC5
    mode = ""
    model_name = ""
    for j, shot in enumerate(shots):
        for i, model in enumerate(models):
            plot_info = PlotInfo(word, shot, model, key_type=key)
            model_name = plot_info.get_model()
            plot_info.quantile_accuracies_plot(q_num=quantile_number)
            mode = plot_info.get_mode()
            quantile_bins = plot_info.quantile_bins
            quantiles = quantile_bins['extrap_quantiles']
            accuracies = quantile_bins['extrap_accuracies']
            h = quantile_bins['extrap_h']
            plt.fill_between(quantiles, accuracies - h , accuracies + h, color=COLORS[i % len(COLORS)], alpha=ALPHA)
            plt.plot(quantiles, accuracies, lw=1, 
                     ls=LINESTYLES[j % len(LINESTYLES)],
                     color=COLORS[i % len(COLORS)],
                     label=f"k={shot}, LM={plot_info.get_model()}")
    plt.legend(loc='upper right', bbox_to_anchor=(0.4, 1))
    plt.xlabel(f'Frequency - ({key})')
    plt.ylabel('Avg. Accuracy')
    plt.ylim(0, 1)
    sns.despine()
    plt.savefig(
        f'./figures4/key_{key}_mode_{mode}_some_shots__all_models_interp.pdf',
        format='pdf',
        dpi=500,
        bbox_inches='tight'
    )
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()

def save_interpolation_lines_plot_for_shots(word: str, models: List[str], shots: List[int], key: str, quantile_number: int = 10, show_plot: bool = False):
    plt.xscale('log')
    mode = ""
    model_name = ""
    for j, shot in enumerate(shots):
        for i, model in enumerate(models):
            plot_info = PlotInfo(word, shot, model, key_type=key)
            model_name = plot_info.get_model()
            plot_info.quantile_accuracies_plot(q_num=quantile_number)
            mode = plot_info.get_mode()
            quantile_bins = plot_info.quantile_bins
            quantiles = quantile_bins['extrap_quantiles']
            accuracies = quantile_bins['extrap_accuracies']
            h = quantile_bins['extrap_h']
            plt.fill_between(quantiles, accuracies - h , accuracies + h, color=COLORS[j % len(COLORS)], alpha=ALPHA)
            plt.plot(quantiles, accuracies, lw=1,
                     ls=LINESTYLES[j % len(LINESTYLES)],
                     color=COLORS[j % len(COLORS)],
                     label=f"k={shot}")
    plt.legend(loc='upper right', bbox_to_anchor=(0.4, 1))
    plt.xlabel(f'Frequency')
    plt.ylabel('Avg. Accuracy')
    plt.savefig(
        f'./figures4/key_{key}_mode_{mode}_all_shots__all_{model[0]}_interp.pdf',
        format='pdf',
        dpi=500,
        bbox_inches='tight'
    )
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()

def save_lines_plot_for_models_shots(word: str, models: List[str], shots: List[int], key_type: str,
                                                         show_plot: bool = False, q_num=10):
    markers = ['*', '^', 'o', 's', '>']
    mode = ""
    model_names = []
    for j, shot in enumerate(shots):
        accs = []
        errors_lower = []
        errors_upper = []
        x_locs = []
        for i, model in enumerate(models):
            # Creat PlotInfo
            plot_info = PlotInfo(word, shot, model, key_type=key_type) \
                .quantile_accuracies_plot(q_num=q_num) \
                .calculate_accuracy_all()
            mode = plot_info.get_mode()
            q_1 = plot_info.quantile_bins['accuracies'][0]
            q_n = plot_info.quantile_bins['accuracies'][-1]
            acc = plot_info.accuracy_all
            # error_bars = [acc-q_1, q_n-acc]
            errors_lower.append(acc-q_1)
            errors_upper.append(q_n-acc)
            x_locs.append(i)
            accs.append(acc)
            # errors.append(error_bars)
            if i>0:
                label=""
            else:
                label= f"shots={shot}"
            a = plt.errorbar(x=i+(j/4), y=acc , yerr=[[acc-q_1], [q_n-acc]], marker=markers[j%len(markers)],
                     capsize=3, label=label, color=COLORS[j % len(COLORS)], lw=3)
            # a[-1][0].set_linestyle(LINESTYLES[j % len(LINESTYLES)])
            model_names.append(plot_info.get_model())
            
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False)  # ticks along the top edge are off
    
    # NOTE(XXX): Slicing model names 'cause I am too lazy to switch the for loops.
    plt.xticks(ticks=[0.25,1.25,2.25], labels=model_names[:3])
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
    plt.ylim(0, 1.05)
#     plt.xlabel(x_label_text)
    plt.ylabel('Avg. Accuracy')
    plt.savefig(f'./figures4/key_{key_type}_linebar_mode_{mode}_some_shots__all_models.pdf', format='pdf', dpi=500, bbox_inches='tight')
    sns.despine()
    if show_plot:
        plt.show()
    plt.close()


def get_farthest_points(word: str,
                        shot: int,
                        model: str,
                        key_type: str,
                        index_filter: Callable[[List[int]], bool] = None,
                        index_score: Callable[[List[int], List[int]], float] = None,
                        freq_score: Callable[[float, float], float] = None,
                        acc_score: Callable[[float, float], float] = None) -> List:
    if freq_score is None:
        freq_score = lambda f_i, f_j: (f_i - f_j)**3
    if acc_score is None:
        acc_score = lambda a_i, a_j: (a_i - a_j)
    index_filter = default_index_filter if index_filter is None else index_filter
    index_score = default_index_score if index_score is None else index_score

    plot_info = PlotInfo(word, shot, model, key_type=key_type)
    is_timeunit = not (word in NO_WORD_KEYS)

    my_sorted_list = []
    print("Size:", len(plot_info.aggregated_data_by_key))
    for i, row_i in plot_info.aggregated_data_by_key.iterrows():
        list_i = convert_any(i, is_timeunit)
        if not index_filter(list_i):
            continue
        f_i = row_i[plot_info.frequency_value_column]
        a_i = row_i['is_correct']
        for j, row_j in plot_info.aggregated_data_by_key.iterrows():
            list_j = convert_any(j, is_timeunit)
            if not index_filter(list_j):
                continue
            if i >= j:
                continue
            f_j = row_j[plot_info.frequency_value_column]
            a_j = row_j['is_correct']
            score = freq_score(f_i,f_j)* acc_score(a_i, a_j) / index_score(list_i, list_j)
            my_sorted_list.append((score, i, j, abs(np.log10(f_i)-np.log10(f_j)), a_i, a_j))
    my_sorted_list.sort(key=lambda x: -x[0])
    return my_sorted_list
