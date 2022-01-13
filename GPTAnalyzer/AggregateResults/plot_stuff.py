import pandas as pd
import matplotlib.pyplot as plt


FREQ_NUM=1
WORD='mult' #mult, #plus or time_units
WORD_FLAG = False
TOP_FREQ = 200
MODEL = 'gptj' #can be 'gptjsmall' 'gptjlarge'
SHOTS = 8 # 2

if WORD in ['mult', 'plus']:
    file_name = f'/home/yrazeghi/PHD/TYM/GPTAnalyzer/results/num{FREQ_NUM}_{WORD}_1to50_top{TOP_FREQ}_{MODEL}_{SHOTS}shots_5seeds_results.csv'
else:
    file_name = f'/home/yrazeghi/PHD/TYM/GPTAnalyzer/results/num{FREQ_NUM}_{WORD}_top{TOP_FREQ}_{MODEL}_{SHOTS}shots_5seeds_results.csv'

data_file = pd.read_csv(file_name)

print(data_file.columns)
print(data_file.iloc[0])


data_file.replace(True, 1, inplace=True)
data_file.replace(False,0, inplace=True)


aggregated_by_key = data_file.groupby('testcase.data_point.frequency_data.key')['is_correct', 'testcase.data_point.frequency_data.frequency'].mean()
aggregated_by_key.plot(kind='scatter', x='testcase.data_point.frequency_data.frequency', y='is_correct')
spearman_correlation = aggregated_by_key.corr(method='spearman')
sp_corr = spearman_correlation.loc['is_correct', 'testcase.data_point.frequency_data.frequency']
# pr_corr = aggregated_by_key.corr(method='pearson').loc['is_correct', 'testcase.data_point.frequency_data.frequency']  # this is a bug should do in log space

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
          )
plt.xlabel('frequency')
plt.ylabel('accuracy')
plt.tight_layout()
plt.subplots_adjust(top=0.75 )

plt.savefig(f'./figures/{MODE}_{SHOTS}shots_{model_name}.pdf', format='pdf', dpi=500)
plt.show()


