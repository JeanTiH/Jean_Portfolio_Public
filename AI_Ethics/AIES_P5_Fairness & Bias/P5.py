""""""
"""OMSCS2024Spring-P5: Fairness & Bias

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)  # None means unlimited
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend, or another appropriate backend
import matplotlib.pyplot as plt

def plot_DI(DI):
    plt.figure(figsize=(10, 6))

    plt.plot([0, 5], [1, 1], color='black', linestyle='-', linewidth=1)
    plt.plot([0, 5], [1.5, 1.5], color='grey', linestyle='--', linewidth=1)
    plt.plot([0, 5], [0.5, 0.5], color='grey', linestyle='--', linewidth=1)
    plt.plot([0, 5], [0, 0], color='grey', linestyle='--', linewidth=1)
    plt.fill_between([0, 5], 0.8, 1.2, color='lightgrey', alpha=0.5)
#    plt.fill_between([0, 5], 0, 0.8, color='mistyrose', alpha=0.5)

    x_range = [2.5, 4]
    x_text = sum(x_range) / 2

    # Using fill_between to create a 'bar'
    plt.fill_between(x_range, 0, DI, color='dimgray', step="pre")
    # Add the DI value on top of the 'bar'
    plt.text(x_text, DI, f'{DI:.2f}', ha='center', va='bottom', fontsize=14)

    plt.text(x=5.5, y=0.95, s='Fair', verticalalignment='bottom', horizontalalignment='right', color='black',
             fontsize=14)
    plt.text(x=5.5, y=0.35, s='Bias', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=14)
    plt.text(x=6, y=1.1, s='Privileged Group: Male', verticalalignment='bottom', horizontalalignment='left',
             color='black', fontsize=14)
    plt.text(x=6, y=0.9, s='Unprivileged Group: Female', verticalalignment='bottom', horizontalalignment='left',
             color='black', fontsize=14)

    plt.xlim(0, 10)
    plt.xticks([])
    plt.ylim(-0.2, 1.7)
    plt.yticks([0, 0.5, 1, 1.5], fontsize=12)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.title('Disparate Impact (Protected Attribute: Sex)', fontsize=16)
    plt.grid(False)

    plt.savefig('5.2_DI.png', bbox_inches='tight')
    plt.close()

def plot_SPD(SPD):
    plt.figure(figsize=(10, 6))

    plt.plot([0, 5], [0, 0], color='black', linestyle='-', linewidth=1)
    plt.plot([0, 5], [-1, -1], color='grey', linestyle='--', linewidth=1)
    plt.plot([0, 5], [-0.5, -0.5], color='grey', linestyle='--', linewidth=1)
    plt.plot([0, 5], [0.5, 0.5], color='grey', linestyle='--', linewidth=1)
    plt.plot([0, 5], [1, 1], color='grey', linestyle='--', linewidth=1)
    plt.fill_between([0, 5], -0.1, 0.1, color='lightgrey', alpha=0.5)

    x_range = [2.5, 4]
    x_text = sum(x_range) / 2

    # Using fill_between to create a 'bar'
    plt.fill_between(x_range, 0, SPD, color='dimgray', step="pre")
    # Add the SPD value on top of the 'bar'
    plt.text(x_text, SPD, f'{SPD:.2f}', ha='center', va='bottom', fontsize=14)

    plt.text(x=5.5, y=-0.05, s='Fair', verticalalignment='bottom', horizontalalignment='right', color='black',
             fontsize=14)
    plt.text(x=5.5, y=-0.6, s='Bias', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=14)
    plt.text(x=6, y=0.1, s='Privileged Group: Male', verticalalignment='bottom', horizontalalignment='left',
             color='black', fontsize=14)
    plt.text(x=6, y=-0.1, s='Unprivileged Group: Female', verticalalignment='bottom', horizontalalignment='left',
             color='black', fontsize=14)

    plt.xlim(0, 10)
    plt.xticks([])
    plt.ylim(-1.1, 1.1)
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=12)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    plt.title('Statistical Parity Difference (Protected Attribute: Sex)', fontsize=16)
    plt.grid(False)

    plt.savefig('5.2_SPD.png', bbox_inches='tight')
    plt.close()
'''
    Main Code 
'''
filename = 'default of credit card clients.xls'
df = pd.read_excel(filename)
df = df.iloc[1:, :] # Drop the second column, which is the instruction
print('Original Data', df.shape)

print('Y=0', (df['Y'] == 0).sum())
print('Y=1', (df['Y'] == 1).sum())
print((df['Y'] == 0).sum() + (df['Y'] == 1).sum())

# Split data into half, drop the first column 'ID'
X = df.iloc[:, 1:24]  # Columns 2-24 as features
y = df.iloc[:, -1]    # The last column as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
print('df_train', df_train.shape)
print('df_test', df_test.shape)

print('------------------------------------')
print('             Step 3.5               ')
print('------------------------------------')
male_num = (df_train['X2'] == 1).sum()
female_num = (df_train['X2'] == 2).sum()
print('Traing Set, male:   ', male_num)
print('Traing Set, female: ',female_num)

male_num = (df_test['X2'] == 1).sum()
female_num = (df_test['X2'] == 2).sum()
print('Testing Set, male:  ', male_num)
print('Testing Set, female:',female_num)

print('------------------------------------')
print('              Step 4                ')
print('------------------------------------')

df_train['Credit'] = 0
df_train.loc[df_train['Y'] == 0, 'Credit'] += 40
df_train.loc[df_train['X1'] > 500000, 'Credit'] += 30
df_train.loc[(df_train['X1'] > 300000) & (df_train['X1'] <= 500000), 'Credit'] += 20
df_train.loc[(df_train['X1'] > 150000) & (df_train['X1'] <= 300000), 'Credit'] += 10
df_train.loc[df_train['X3'] < 3, 'Credit'] += 10
non_positive_condition = (df_train.loc[:, 'X6':'X11'] <= 0).all(axis=1)
df_train.loc[non_positive_condition, 'Credit'] += 20
df_train.to_csv('df_train.csv', index=False)

print('**************')
print('   Step 4.1   ')
print('**************')
threshold_str = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
threshold = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]

credit_dict = {}
for i in range(len(threshold_str)):
    credit_dict[threshold_str[i]] = ((df_train['Credit'] >= threshold[i]) & (df_train['Credit'] < threshold[i+1])).sum()
print('Credit Distribution', credit_dict)

plt.figure(figsize=(10, 6))
plt.bar(credit_dict.keys(), credit_dict.values())
plt.xlabel('Creditworthiness', fontsize=16)
plt.ylabel('Members', fontsize=16)
plt.title('Members of Creditworthiness', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('4.1.png', bbox_inches='tight')
plt.close()

print('**************')
print('   Step 4.2   ')
print('**************')
# Y=0 and Approval=0 approve the loan
# Y=1 and Approval=1 deny the loan
profits = []
print((df_train['Y'] == 0).sum())
print((df_train['Y'] == 1).sum())

for i in range(10):
    profit = 0
    df_train['Approval'] = 1
    df_train.loc[(df_train['Credit'] >= threshold[i]), 'Approval'] = 0
    profit += ((df_train['Y'] == 0) & (df_train['Approval'] == 0)).sum() * 10
    profit -= ((df_train['Y'] == 0) & (df_train['Approval'] == 1)).sum() * 5
    profit -= ((df_train['Y'] == 1) & (df_train['Approval'] == 0)).sum() * 3
    profits.append(profit)

max_value = max(profits)  # Find the maximum value in the list
max_value_index = profits.index(max_value)
threshold_value = threshold[max_value_index]
print('Profits:', profits)
print('Max Profit Threshold:', threshold_value)

# Modify 'Approval' based on the threshold_value
df_train['Approval'] = 1
df_train.loc[(df_train['Credit'] >= threshold_value), 'Approval'] = 0

print('**************')
print('   Step 4.3   ')
print('**************')
bad_credit_risk = {k: v for k, v in credit_dict.items() if k in threshold_str[:max_value_index]}
good_credit_risk = {k: v for k, v in credit_dict.items() if k not in threshold_str[:max_value_index]}

plt.figure(figsize=(10, 6))
plt.bar(bad_credit_risk.keys(), bad_credit_risk.values(), color='darkblue', label='Bad Credit Risk')
plt.bar(good_credit_risk.keys(), good_credit_risk.values(), color='#86c1ea', label='Good Credit Risk')

plt.xlabel('Creditworthiness', fontsize=16)
plt.ylabel('Members', fontsize=16)
plt.title('Members of Creditworthiness', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

x_positions = range(len(credit_dict.keys()))  # Get positions for each key in the dict
line_position_1 = list(credit_dict.keys()).index(threshold_str[max_value_index-1])
line_position_2 = list(credit_dict.keys()).index(threshold_str[max_value_index])
vertical_line_position = (x_positions[line_position_1] + x_positions[line_position_2]) / 2
plt.axvline(x=vertical_line_position, color='red', linestyle='-')
plt.text(vertical_line_position + 0.2, max(credit_dict.values()) / 1, 'Threshold='+str(threshold[max_value_index]), fontsize=16, color='red')

plt.savefig('4.3.png', bbox_inches='tight')
plt.close()
print('**************')
print('   Step 4.4   ')
print('**************')
Male_appr = df_train[(df_train['X2'] == 1) & (df_train['Approval'] == 0)].shape[0]
Male_deny = df_train[(df_train['X2'] == 1) & (df_train['Approval'] == 1)].shape[0]
Female_appr = df_train[(df_train['X2'] == 2) & (df_train['Approval'] == 0)].shape[0]
Female_deny = df_train[(df_train['X2'] == 2) & (df_train['Approval'] == 1)].shape[0]
print('Male:   Approval', Male_appr, '  Denial', Male_deny)
print('Female: Approval', Female_appr, '  Denial', Female_deny)

print('------------------------------------')
print('              Step 5                ')
print('------------------------------------')
Male_total = df_train[(df_train['X2'] == 1)].shape[0]
Female_total = df_train[(df_train['X2'] == 2)].shape[0]
print('Rate of Positive Outcome for Male  :', f"{Male_appr/(Male_total):.0%}")
print('Rate of Positive Outcome for FeMale:', f"{Female_appr/(Female_total):.0%}")
print()
# Metric1: Disparate Impact (DI)
DI = (Female_appr/Female_total) / (Male_appr/Male_total)
DI_flip = (Male_appr/Male_total)/(Female_appr/Female_total)
plot_DI(DI)
print('The Disparate Impact:', DI)
print('DI:', DI, ', DI_flip:', DI_flip, ', Diff: ', DI-DI_flip)
print()
# Metric2: Statistical Parity Difference (SPD)
SPD = Female_appr/Female_total - Male_appr/Male_total
SPD_flip = Male_appr/Male_total - Female_appr/Female_total
plot_SPD(SPD)
print('The Statistical Parity Difference:', SPD)
print('SPD:', SPD, ', SPD_flip:', SPD_flip, ', Diff: ', SPD-SPD_flip)

print('------------------------------------')
print('              Step 6                ')
print('------------------------------------')
thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for male_threshold in thresholds:
    df_train['Mitigate'] = 1  # Reset Mitigate for all before checking male threshold
    df_train.loc[(df_train['X2'] == 1) & (df_train['Credit'] >= male_threshold), 'Mitigate'] = 0
    Male_appr = df_train[(df_train['X2'] == 1) & (df_train['Mitigate'] == 0)].shape[0]

    for female_threshold in thresholds:
        # Reset Mitigate for females for each female threshold check
        df_train['Mitigate'] = 1
        df_train.loc[(df_train['X2'] == 1) & (df_train['Credit'] >= male_threshold), 'Mitigate'] = 0  # Reapply male threshold
        df_train.loc[(df_train['X2'] == 2) & (df_train['Credit'] >= female_threshold), 'Mitigate'] = 0  # Apply current female threshold
        Female_appr = df_train[(df_train['X2'] == 2) & (df_train['Mitigate'] == 0)].shape[0]

        DI = (Female_appr / Female_total) / (Male_appr / Male_total)
        DI_flip = (Male_appr / Male_total) / (Female_appr / Female_total)
        SPD = Female_appr / Female_total - Male_appr / Male_total
        SPD_flip = Male_appr / Male_total - Female_appr / Female_total

        profit = 0
        df_train['Mitigate_approval'] = 1
        profit += ((df_train['Y'] == 0) & (df_train['Mitigate'] == 0)).sum() * 10
        profit -= ((df_train['Y'] == 0) & (df_train['Mitigate'] == 1)).sum() * 5
        profit -= ((df_train['Y'] == 1) & (df_train['Mitigate'] == 0)).sum() * 3

        if (DI > 0.96 and DI < 1.04 ) or (SPD > -0.03 and SPD < 0.03):
            print('Male_Threshold: ' + str(male_threshold) + ', Female_Threshold: ' + str(female_threshold) +
                ' Male_Appr: ', str(Male_appr) + ' Male_Denial: ', str(Male_total-Male_appr) +
                ' Female_Appr: ', str(Female_appr) + ' Female_Denial: ', str(Female_total - Female_appr) +
                ', DI_diff: ' + f"{DI-DI_flip:.3f}" + ', SPD_diff: ' + f"{SPD-SPD_flip:.3f}" + ', Profit: ' + str(profit))

print('**************')
print('   Step 6.3   ')
print('**************')
# Privileged group
max_value_index = 2
credit_dict = {}
for i in range(len(threshold_str)):
    credit_dict[threshold_str[i]] = ((df_train['X2'] == 1) & (df_train['Credit'] >= threshold[i]) & (df_train['Credit'] < threshold[i+1])).sum()

bad_credit_risk = {k: v for k, v in credit_dict.items() if k in threshold_str[:max_value_index]}
good_credit_risk = {k: v for k, v in credit_dict.items() if k not in threshold_str[:max_value_index]}

plt.figure(figsize=(10, 6))
plt.bar(bad_credit_risk.keys(), bad_credit_risk.values(), color='darkblue', label='Bad Credit Risk')
plt.bar(good_credit_risk.keys(), good_credit_risk.values(), color='#86c1ea', label='Good Credit Risk')

plt.xlabel('Creditworthiness', fontsize=16)
plt.ylabel('Members', fontsize=16)
plt.title('Members of Creditworthiness for Privileged Group', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

x_positions = range(len(credit_dict.keys()))  # Get positions for each key in the dict
line_position_1 = list(credit_dict.keys()).index(threshold_str[max_value_index-1])
line_position_2 = list(credit_dict.keys()).index(threshold_str[max_value_index])
vertical_line_position = (x_positions[line_position_1] + x_positions[line_position_2]) / 2
plt.axvline(x=vertical_line_position, color='red', linestyle='-')
plt.text(vertical_line_position + 0.2, max(credit_dict.values()) / 1, 'Threshold='+str(threshold[max_value_index]), fontsize=16, color='red')

plt.savefig('6.3_male.png', bbox_inches='tight')
plt.close()

# Unprivileged group
max_value_index = 3
credit_dict = {}
for i in range(len(threshold_str)):
    credit_dict[threshold_str[i]] = ((df_train['X2'] == 2) & (df_train['Credit'] >= threshold[i]) & (df_train['Credit'] < threshold[i+1])).sum()

bad_credit_risk = {k: v for k, v in credit_dict.items() if k in threshold_str[:max_value_index]}
good_credit_risk = {k: v for k, v in credit_dict.items() if k not in threshold_str[:max_value_index]}

plt.figure(figsize=(10, 6))
plt.bar(bad_credit_risk.keys(), bad_credit_risk.values(), color='darkblue', label='Bad Credit Risk')
plt.bar(good_credit_risk.keys(), good_credit_risk.values(), color='#86c1ea', label='Good Credit Risk')

plt.xlabel('Creditworthiness', fontsize=16)
plt.ylabel('Members', fontsize=16)
plt.title('Members of Creditworthiness for Unprivileged Group', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()

x_positions = range(len(credit_dict.keys()))  # Get positions for each key in the dict
line_position_1 = list(credit_dict.keys()).index(threshold_str[max_value_index-1])
line_position_2 = list(credit_dict.keys()).index(threshold_str[max_value_index])
vertical_line_position = (x_positions[line_position_1] + x_positions[line_position_2]) / 2
plt.axvline(x=vertical_line_position, color='red', linestyle='-')
plt.text(vertical_line_position + 0.2, max(credit_dict.values()) / 1, 'Threshold='+str(threshold[max_value_index]), fontsize=16, color='red')

plt.savefig('6.3_female.png', bbox_inches='tight')
plt.close()