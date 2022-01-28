# should read from utils
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from logistic_regression import logistic
from scipy.stats.mstats import mquantiles
import seaborn as sns


result = ""
digit_limit = 2
num_2_frequency_dic = {}
start = time.time()
with open("num_1_counting.txt") as f:
    for line in f.readlines():
        key, frequency = eval(line)
        if len(key[0]) <= digit_limit or len(key[1]) <= digit_limit:
            num_2_frequency_dic[key] = frequency

print(
    f"done with readint the frequency here is read and it took {time.time() - start} seconds {type(num_2_frequency_dic)}")

##############################################################################################

FREQ_NUM = 1  # can be 1 2 3
WORD = 'mult'  # mult, #plus
WORD_FLAG = False
TOP_FREQ = 200
MODEL = 'gptj'  # can be 'gptjsmall' 'gptjlarge'
SHOTS = 4

# file_name = f'/home/XXXX/PHD/TYM/GPTAnalyzer/results/num{FREQ_NUM}_{WORD}_top{TOP_FREQ}_{MODEL}_{SHOTS}shots_5seeds_results.csv'
file_name = f'/home/XXXX/PHD/TYM/GPTAnalyzer/results/num{FREQ_NUM}_{WORD}_1to50_top{TOP_FREQ}_{MODEL}_{SHOTS}shots_5seeds_results.csv'
data_file = pd.read_csv(file_name)
not_found = 0
for i, row in data_file.iterrows():
    num_2_key = ""
    num_2_frequency = 0
    min_val = min(str(row['testcase.data_point.frequency_data.x']), str(row['testcase.data_point.frequency_data.z']))
    max_val = max(str(row['testcase.data_point.frequency_data.x']), str(row['testcase.data_point.frequency_data.z']))
    tuple1 = (min_val, max_val)
    tuple2 = (max_val, min_val)
    if tuple1 in num_2_frequency_dic.keys() and tuple2 in num_2_frequency_dic.keys() and not (tuple1 == tuple2):
        print(f"something is very wrong!!!!!!!!!!! tuple1: {tuple1} tuple2: {tuple2}")
    if tuple1 in num_2_frequency_dic.keys():
        num_2_key = str(tuple1)
        num_2_frequency = num_2_frequency_dic[tuple1]
    elif tuple2 in num_2_frequency_dic.keys():
        num_2_key = str(tuple2)
        num_2_frequency = num_2_frequency_dic[tuple2]
    else:
        num_2_key = str(tuple1)
        not_found = not_found + 1
    data_file.at[i, 'num_2_key'] = num_2_key
    data_file.at[i, 'num_2_frequency'] = num_2_frequency
print(f"not found number is {not_found}")

FREQ = 'log_num_2_frequency'
data_file['log_num_2_frequency'] = np.log(data_file['num_2_frequency'])

aggregated_by_key = data_file.groupby('num_2_key')['is_correct', FREQ, 'num_2_frequency'].mean()
print(aggregated_by_key.head())
# sns.scatterplot(data=aggregated_by_key, x='num_2_frequency', y='is_correct', x_jitter=True, y_jitter=True)
# sns.regplot(data=aggregated_by_key, x='num_2_frequency', y='is_correct', logx=True, scatter=True, fit_reg=True)
# sns.regplot(data=aggregated_by_key, x='num_2_frequency', y='is_correct', scatter = True ,logistic=True)
aggregated_by_key.plot(kind='scatter', x='num_2_frequency', y='is_correct')
spearman_correlation = aggregated_by_key.corr(method='spearman')
sp_corr = spearman_correlation.loc['is_correct', FREQ]
pr_corr = aggregated_by_key.corr(method='pearson').loc['is_correct', FREQ]

if MODEL == 'gptjsmall':
    model_name = 'GPT-NEO-1.3B'
elif MODEL == 'gptjlarge':
    model_name = 'GPT-NEO-2.7B'
elif MODEL == 'gptj':
    model_name = 'GPT-J-6B'
if WORD == 'mult' and WORD_FLAG==False:
    MODE = 'Arithmetics-Multiplication'
elif WORD == 'plus' and WORD_FLAG==False:
    MODE = 'Arithmetics-Adding'
else:
    MODE = f'Time-Coversion-From{WORD}'
plt.xscale('log')
plt.title(f'mode:    {MODE}\n' 
          f'shots:            {SHOTS}\n'
          f'model:   {model_name}\n'
          f'spearman corr:   {sp_corr:.3f}\n'
          f'pearson corr:    {pr_corr:.3f}', loc='left'
          )
plt.xlabel('frequency of (x, z)')
plt.ylabel('accuracy')
plt.tight_layout()
plt.subplots_adjust(top=0.75 )
# plt.show()
##############################################################################################
print("hear I am")
temperature = data_file[FREQ]
D = data_file['is_correct']  # defect or not?

#notice the`value` here. We explain why below.

with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=0.001, testval=0)
    p = pm.Deterministic("p", 1.0/(1. + tt.exp(beta*temperature + alpha)))

with model:
    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)

    # Mysterious code to be explained in Chapter 3
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace["alpha"][:, None]  # best to make them 1d
beta_samples = burned_trace["beta"][:, None]

t = np.linspace(temperature.min() - 5, temperature.max()+5, 100)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)

mean_prob_t = p_t.mean(axis=0)

t = np.exp(t)

qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7,
                 color="#7A68A6")

plt.plot(t[:, 0], qs[0], label="95% CI", color="#7A68A6", alpha=0.7)

plt.plot(t, mean_prob_t, lw=1, ls="--", color="k",
         label="average posterior \nprobability of correctness")

plt.savefig(f'./log_plots/{MODE}_{SHOTS}shots_{model_name}_logreg.png', format='png', dpi=500)
plt.show()

