""""""
"""OMSCS2024Spring-P3: Toxicity Analysis		  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14, 'axes.labelsize': 14})

'''
#################################################
            Step1 Data preprocessing
#################################################
'''
def preprocess(filename):
    data_raw = pd.read_csv(filename)
    print('Raw   Data Shape: ', data_raw.shape)
    input('pause')
    #data_raw = data_raw[(data_raw['age'] != 'Unk')]
    #print('Clean Data Shape: ', data_raw.shape)
    #print('---------------------------')

    # -------------------------------------------
    # 1. Group Manner_of_death (11 to 6 groups)
    # --------------------------------------------
    # 1. Homicide: Homicide Willful (Other Inmate), Homicide Justified (Law Enforcement Staff), Homicide Willful (Law Enforcement Staff), Homicide Justified (Other Inmate)
    # 2. Other: Execution, Pending Investigation, Other
    # 3. Suicide
    # 4. Natural
    # 5. Accidental
    # 6. Cannot be Determined (CD)
    # -------------------------------------------
    Homicides = ['Homicide Willful (Other Inmate)', 'Homicide Justified (Law Enforcement Staff)', 'Homicide Willful (Law Enforcement Staff)', 'Homicide Justified (Other Inmate)']
    Others = ['Execution', 'Pending Investigation', 'Other']
    data_raw['manner_of_death_new'] = ''
    data_raw.loc[data_raw['manner_of_death'].isin(Homicides), 'manner_of_death_new'] = 'Homicide'
    data_raw.loc[data_raw['manner_of_death'].isin(Others), 'manner_of_death_new'] = 'Other'
    data_raw.loc[data_raw['manner_of_death'].eq('Suicide'), 'manner_of_death_new'] = 'Suicide'
    data_raw.loc[data_raw['manner_of_death'].eq('Natural'), 'manner_of_death_new'] = 'Natural'
    data_raw.loc[data_raw['manner_of_death'].eq('Accidental'), 'manner_of_death_new'] = 'Accidental'
    data_raw.loc[data_raw['manner_of_death'].eq('Cannot be Determined'), 'manner_of_death_new'] = 'CD'

    homicide = data_raw['manner_of_death_new'].value_counts().get('Homicide', 0)
    other = data_raw['manner_of_death_new'].value_counts().get('Other', 0)
    suicide = data_raw['manner_of_death_new'].value_counts().get('Suicide', 0)
    natural = data_raw['manner_of_death_new'].value_counts().get('Natural', 0)
    accidental = data_raw['manner_of_death_new'].value_counts().get('Accidental', 0)
    undetermined = data_raw['manner_of_death_new'].value_counts().get('CD', 0)
    print('Record of different manner_of_death: ', homicide, other, suicide, natural, accidental, undetermined, homicide+other+suicide+natural+accidental+undetermined)

    # -------------------------------------------
    # 2. Group Custody_status (8 to 5 groups)
    # --------------------------------------------
    # 1. Other: In Transit, Process of Arrest, Out to Court, Other
    # 2. Sentenced
    # 3. Awaiting Booking (AB)
    # 4. Booked - Awaiting Trial (B-AT)
    # 5. Booked - No Charges Filed (B-NCF)
    # -------------------------------------------
    Others = ['In Transit', 'Process of Arrest', 'Out to Court', 'Other']
    data_raw['custody_status_new'] = ''
    data_raw.loc[data_raw['custody_status'].isin(Others), 'custody_status_new'] = 'Other'
    data_raw.loc[data_raw['custody_status'].eq('Sentenced'), 'custody_status_new'] = 'Sentenced'
    data_raw.loc[data_raw['custody_status'].eq('Awaiting Booking'), 'custody_status_new'] = 'AB'
    data_raw.loc[data_raw['custody_status'].eq('Booked - Awaiting Trial'), 'custody_status_new'] = 'B-AT'
    data_raw.loc[data_raw['custody_status'].eq('Booked - No Charges Filed'), 'custody_status_new'] = 'B-NCF'

    other = data_raw['custody_status_new'].value_counts().get('Other', 0)
    sentenced = data_raw['custody_status_new'].value_counts().get('Sentenced', 0)
    awaiting = data_raw['custody_status_new'].value_counts().get('AB', 0)
    trial = data_raw['custody_status_new'].value_counts().get('B-AT', 0)
    nocharge = data_raw['custody_status_new'].value_counts().get('B-NCF', 0)
    print('Record of different custody_status:  ', other, sentenced, awaiting, trial, nocharge, other+sentenced+awaiting+trial+nocharge)

    # -------------------------------------------
    # 3. Group Race (18 to 4 groups)
    # --------------------------------------------
    # 1. Other: Chinese, Vietnamese, Korean, Japanese, Filipino, Cambodian, Laotian, Asian Indian, Other Asian, Samoan, Hawaiian, Guamanian, American Indian, Pacific Islander, Other
    # 2. White
    # 3. Blcak
    # 4. Hispanic
    # -------------------------------------------
    Others = ['Chinese', 'Vietnamese', 'Korean', 'Japanese', 'Filipino', 'Cambodian', 'Laotian', 'Asian Indian',
              'Other Asian', 'Samoan', 'Hawaiian', 'Guamanian', 'American Indian', 'Pacific Islander', 'Other']
    data_raw['race_new'] = ''
    data_raw.loc[data_raw['race'].isin(Others), 'race_new'] = 'Other'
    data_raw.loc[data_raw['race'].eq('White'), 'race_new'] = 'White'
    data_raw.loc[data_raw['race'].eq('Black'), 'race_new'] = 'Black'
    data_raw.loc[data_raw['race'].eq('Hispanic'), 'race_new'] = 'Hispanic'

    white = data_raw['race_new'].value_counts().get('White', 0)
    black = data_raw['race_new'].value_counts().get('Black', 0)
    hispanic = data_raw['race_new'].value_counts().get('Hispanic', 0)
    other = data_raw['race_new'].value_counts().get('Other', 0)
    print('Record of different races:           ', white, black, hispanic, other, white + black + hispanic + other)

    # data_raw[['race', 'race_new']].to_csv('test.csv', index=True)

    # -------------------------------------------
    # 4. Group Age (3 groups)
    # --------------------------------------------
    # 1. Minor: <18
    # 2. Adult: >=18 & < 65
    # 3. Senior: >= 65
    # -------------------------------------------
    data_raw['age'] = data_raw['age'].astype(int)
    data_raw['age_new'] = ''
    data_raw.loc[data_raw['age'].astype(float) < 40, 'age_new'] = 'Under 40'
    data_raw.loc[data_raw['age'].astype(float) >= 40, 'age_new'] = '40 & Above'

    under = data_raw['age_new'].value_counts().get('Under 40', 0)
    above = data_raw['age_new'].value_counts().get('40 & Above', 0)
    print('Record of different ages:            ', under, above, under+above)

    # -------------------------------------------
    # 5. Gender (2 groups)
    # --------------------------------------------
    # 1. Male
    # 2. Female
    # -------------------------------------------
    male = data_raw['gender'].value_counts().get('Male', 0)
    female = data_raw['gender'].value_counts().get('Female', 0)
    print('Record of different genders:         ', male, female, male+female)

    return data_raw

'''
#################################################
            Read Data and Run Experiments
#################################################
'''
if __name__ == "__main__":
    filename = 'toxity_per_attribute.csv'
    data = preprocess(filename)

    '''
    Manner of Death
    '''
    # 1. Age (Under 40, 40 & Above) & Manner_of_death (Homicide, Suicide, Natural, Accidental, Cannot be Determined, Other)
    count_dict = {'Under 40': {}, '40 & Above': {}}
    # Iterate over age and death combinations
    for age in ['Under 40', '40 & Above']:
        for death in ['Homicide', 'Suicide', 'Natural', 'Accidental', 'CD', 'Other']:
            # Filter the DataFrame based on age and death
            subset = data[(data['age_new'] == age) & (data['manner_of_death_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[age][death] = len(subset)
    # Print the counts
    total=0
    for age, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(age, death, count)
            total += count
    print(total)
    print('---------------------------')

    # Extract data for plotting
    ages = list(count_dict.keys())
    manners = list(count_dict['Under 40'].keys())
    counts = np.array([[count_dict[age][manner] for manner in manners] for age in ages])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, age in enumerate(ages):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=age)

    ax.set_xticks(np.arange(len(manners)) + width * (len(ages) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Manner of Death', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Age', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Manner of Death by Age', fontsize='16')
    plt.savefig('manner_of_death_by_age.png', bbox_inches='tight')
    plt.close()

    # 2. Gender (Male, Female) & Manner_of_death (Homicide, Suicide, Natural, Accidental, Cannot be Determined, Other)
    count_dict = {'Male': {}, 'Female': {}}
    # Iterate over gender and death combinations
    for gender in ['Male', 'Female']:
        for death in ['Homicide', 'Suicide', 'Natural', 'Accidental', 'CD', 'Other']:
            # Filter the DataFrame based on gender and death
            subset = data[(data['gender'] == gender) & (data['manner_of_death_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[gender][death] = len(subset)
    # Print the counts
    total = 0
    for gender, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(gender, death, count)
            total += count
    print(total)
    print('---------------------------')

    # Extract data for plotting
    genders = list(count_dict.keys())
    manners = list(count_dict['Male'].keys())
    counts = np.array([[count_dict[gender][manner] for manner in manners] for gender in genders])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, gender in enumerate(genders):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=gender)

    ax.set_xticks(np.arange(len(manners)) + width * (len(genders) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Manner of Death', fontsize='20')
    ax.set_ylabel('Count of Manner of Death', fontsize='20')
    ax.legend(title='Gender', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Manner of Death by Gender', fontsize='16')
    plt.savefig('manner_of_death_by_gender.png', bbox_inches='tight')
    plt.close()

    # 3. Race (White, Black, Hispanic, Other) & Manner_of_death (Homicide, Suicide, Natural, Accidental, Cannot be Determined (CD), Other)
    count_dict = {'White': {}, 'Black': {}, 'Hispanic': {}, 'Other': {}}
    # Iterate over race and death combinations
    for race in ['White', 'Black', 'Hispanic', 'Other']:
        for death in ['Homicide', 'Suicide', 'Natural', 'Accidental', 'CD', 'Other']:
            # Filter the DataFrame based on race and death
            subset = data[(data['race_new'] == race) & (data['manner_of_death_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[race][death] = len(subset)
    # Print the counts
    total = 0
    for race, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(race, death, count)
            total += count
    print(total)

    # Extract data for plotting
    races = list(count_dict.keys())
    manners = list(count_dict['White'].keys())
    counts = np.array([[count_dict[race][manner] for manner in manners] for race in races])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, race in enumerate(races):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=race)

    ax.set_xticks(np.arange(len(manners)) + width * (len(races) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Manner of Death', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Race', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Manner of Death by Race', fontsize='16')
    plt.savefig('manner_of_death_by_race.png', bbox_inches='tight')
    plt.close()

    print('***********************************')

    '''
    Custody Status
    '''
    # 4. Age (Under 40, 40 & Above) & Custody_status (Sentenced, Awaiting Booking (AB), Booked - Awaiting Trial (B-AT), Booked - No Charges Filed (B-NCF),Other)
    count_dict = {'Under 40': {}, '40 & Above': {}}
    # Iterate over age and death combinations
    for age in ['Under 40', '40 & Above']:
        for death in ['Sentenced', 'AB', 'B-AT', 'B-NCF', 'Other']:
            # Filter the DataFrame based on age and death
            subset = data[(data['age_new'] == age) & (data['custody_status_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[age][death] = len(subset)
    # Print the counts
    total = 0
    for age, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(age, death, count)
            total += count
    print(total)
    print('---------------------------')

    # Extract data for plotting
    ages = list(count_dict.keys())
    manners = list(count_dict['Under 40'].keys())
    counts = np.array([[count_dict[age][manner] for manner in manners] for age in ages])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, age in enumerate(ages):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=age)

    ax.set_xticks(np.arange(len(manners)) + width * (len(ages) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Age', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Custody Status by Age', fontsize='16')
    plt.savefig('custody_status_by_age.png', bbox_inches='tight')
    plt.close()
    '''
    # 5.1 Fairness plot
    '''
    # Extracting the values for plotting
    under_ratio_sentenced = count_dict['Under 40']['Sentenced'] / (count_dict['Under 40']['Sentenced'] + count_dict['40 & Above']['Sentenced'])
    above_ratio_sentenced = count_dict['40 & Above']['Sentenced'] / (count_dict['Under 40']['Sentenced'] + count_dict['40 & Above']['Sentenced'])
    print(under_ratio_sentenced, above_ratio_sentenced)

    # Plotting the line
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(['Under 40', '40 & Above'], [under_ratio_sentenced, above_ratio_sentenced], marker='o', linestyle='-')

    plt.ylim(bottom=-100, top=100)
    plt.xlabel('Age', fontsize='20')
    # plt.ylabel('Ratio of Sentenced Death', fontsize='20')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Difference in Sentenced Death Based on Age', fontsize='16')
    plt.savefig('fairness-age.png', bbox_inches='tight')
    plt.close()
    '''
    # 5.2 Bias plot
    '''
    # Extracting the values for plotting
    under_sentenced = count_dict['Under 40']['Sentenced']
    above_sentenced = count_dict['40 & Above']['Sentenced']
    under_non_sentenced = count_dict['Under 40']['AB'] + count_dict['Under 40']['B-AT'] + count_dict['Under 40']['B-NCF'] + count_dict['Under 40']['Other']
    above_non_sentenced = count_dict['40 & Above']['AB'] + count_dict['40 & Above']['B-AT'] + count_dict['40 & Above']['B-NCF'] + count_dict['40 & Above']['Other']
    count_dict = {'Under 40': {'Sentenced': under_sentenced, 'Non-Sentenced': under_non_sentenced},
                  '40 & Above': {'Sentenced': above_sentenced, 'Non-Sentenced': above_non_sentenced}}

    ages = list(count_dict.keys())
    manners = list(count_dict['Under 40'].keys())
    counts = np.array([[count_dict[age][manner] for manner in manners] for age in ages])

    # Plotting the bar
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    bar_positions = np.arange(len(manners))

    for i, age in enumerate(ages):
        bottom = np.sum(counts[:i], axis=0)  # Calculate the bottom position for each set of bars
        ax.bar(bar_positions, counts[i], width=bar_width, label=age, bottom=bottom)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(manners)
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Age', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Difference in Sentenced Death Based on Age', fontsize='16')
    plt.savefig('bias-age.png', bbox_inches='tight')
    plt.close()

    # 5. Gender (Male, Female) & Manner_of_death (Homicide, Suicide, Natural, Accidental, Cannot be Determined, Other)
    count_dict = {'Male': {}, 'Female': {}}
    # Iterate over gender and death combinations
    for gender in ['Male', 'Female']:
        for death in ['Sentenced', 'AB', 'B-AT', 'B-NCF', 'Other']:
            # Filter the DataFrame based on gender and death
            subset = data[(data['gender'] == gender) & (data['custody_status_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[gender][death] = len(subset)
    # Print the counts
    total = 0
    for gender, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(gender, death, count)
            total += count
    print(total)
    print('---------------------------')

    # Extract data for plotting
    genders = list(count_dict.keys())
    manners = list(count_dict['Male'].keys())
    counts = np.array([[count_dict[gender][manner] for manner in manners] for gender in genders])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, gender in enumerate(genders):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=gender)

    ax.set_xticks(np.arange(len(manners)) + width * (len(genders) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Gender', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Custody Status by Gender', fontsize='16')
    plt.savefig('custody_status_by_gender.png', bbox_inches='tight')
    plt.close()
    '''
    # 5.1 Fairness plot
    '''
    # Extracting the values for plotting
    male_ratio_sentenced = count_dict['Male']['Sentenced'] / (count_dict['Male']['Sentenced'] + count_dict['Female']['Sentenced'])
    female_ratio_sentenced = count_dict['Female']['Sentenced'] / (count_dict['Male']['Sentenced'] + count_dict['Female']['Sentenced'])
    #print(male_ratio_sentenced, female_ratio_sentenced)

    # Plotting the line
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(['Male', 'Female'], [male_ratio_sentenced, female_ratio_sentenced], marker='o', linestyle='-')

    plt.ylim(bottom=-45, top=45)
    plt.xlabel('Gender', fontsize='20')
    #plt.ylabel('Ratio of Sentenced Death', fontsize='20')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Difference in Sentenced Death Based on Gender', fontsize='16')
    plt.savefig('fairness-gender.png', bbox_inches='tight')
    plt.close()
    '''
    # 5.2 Bias plot
    '''
    # Extracting the values for plotting
    male_sentenced = count_dict['Male']['Sentenced']
    female_sentenced = count_dict['Female']['Sentenced']
    male_non_sentenced = count_dict['Male']['AB'] + count_dict['Male']['B-AT'] + count_dict['Male']['B-NCF'] + count_dict['Male']['Other']
    female_non_sentenced = count_dict['Female']['AB'] + count_dict['Female']['B-AT'] + count_dict['Female']['B-NCF'] + count_dict['Female']['Other']
    count_dict = {'Male': {'Sentenced Death': male_sentenced, 'Non-Sentenced Death': male_non_sentenced},
                  'Female': {'Sentenced Death': female_sentenced, 'Non-Sentenced Death': female_non_sentenced}}

    genders = list(count_dict.keys())
    manners = list(count_dict['Male'].keys())
    counts = np.array([[count_dict[gender][manner] for manner in manners] for gender in genders])

    # Plotting the bar
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.35
    bar_positions = np.arange(len(manners))

    for i, gender in enumerate(genders):
        bottom = np.sum(counts[:i], axis=0)  # Calculate the bottom position for each set of bars
        ax.bar(bar_positions, counts[i], width=bar_width, label=gender, bottom=bottom)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(manners)
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Gender', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Difference in Sentenced Death Based on Gender', fontsize='16')
    plt.savefig('bias-gender.png', bbox_inches='tight')
    plt.close()

    # 6. Race (White, Black, Hispanic, Other) & Manner_of_death (Homicide, Suicide, Natural, Accidental, Cannot be Determined, Other)
    count_dict = {'White': {}, 'Black': {}, 'Hispanic': {}, 'Other': {}}
    # Iterate over race and death combinations
    for race in ['White', 'Black', 'Hispanic', 'Other']:
        for death in ['Sentenced', 'AB', 'B-AT', 'B-NCF', 'Other']:
            # Filter the DataFrame based on race and death
            subset = data[(data['race_new'] == race) & (data['custody_status_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[race][death] = len(subset)
    # Print the counts
    total = 0
    for race, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(race, death, count)
            total += count
    print(total)

    # Extract data for plotting
    races = list(count_dict.keys())
    manners = list(count_dict['White'].keys())
    counts = np.array([[count_dict[race][manner] for manner in manners] for race in races])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, race in enumerate(races):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=race)

    ax.set_xticks(np.arange(len(manners)) + width * (len(races) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Race', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Custody Status by Race', fontsize='16')
    plt.savefig('custody_status_by_race.png', bbox_inches='tight')
    plt.close()

    '''
    Step 5
    '''
    # age
    data.loc[data['age_new'] == 'Under 40', 'age_num'] = 1
    data.loc[data['age_new'] == '40 & Above', 'age_num'] = 2

    mean_age = data['age_num'].mean()
    median_age = data['age_num'].median()
    mode_age = data['age_num'].mode()
    print('************************')
    print('Step 5 starts here')
    print('************************')
    print("Original Mean Age:", mean_age)
    print("Original Median Age:", median_age)
    print("Original Mode Age:", mode_age)

    # Randomly sample 50% of the data
    reduced_data = data.sample(frac=0.5, random_state=42)
    mean_age = reduced_data['age_num'].mean()
    median_age = reduced_data['age_num'].median()
    mode_age = reduced_data['age_num'].mode()
    print("Reduce Mean Age:", mean_age)
    print("Reduce Median Age:", median_age)
    print("Reduce Mode Age:", mode_age)

    '''
    Step 6
    '''
    print('************************')
    print('Step 6 starts here')
    print('************************')
    # Age (Under 40, 40 & Above) & Custody_status (Sentenced, Awaiting Booking (AB), Booked - Awaiting Trial (B-AT), Booked - No Charges Filed (B-NCF),Other)
    count_dict = {'Under 40': {}, '40 & Above': {}}
    # Iterate over age and death combinations
    for age in ['Under 40', '40 & Above']:
        for death in ['Sentenced', 'AB', 'B-AT', 'B-NCF', 'Other']:
            # Filter the DataFrame based on age and death
            subset = reduced_data[(reduced_data['age_new'] == age) & (reduced_data['custody_status_new'] == death)]
            # Get the count for the current combination and store it in the dictionary
            count_dict[age][death] = len(subset)
    # Print the counts
    total = 0
    for age, death_counts in count_dict.items():
        for death, count in death_counts.items():
            print(age, death, count)
            total += count
    print(total)
    print('---------------------------')

    # Extract data for plotting
    ages = list(count_dict.keys())
    manners = list(count_dict['Under 40'].keys())
    counts = np.array([[count_dict[age][manner] for manner in manners] for age in ages])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    width = 0.15
    for i, age in enumerate(ages):
        ax.bar(np.arange(len(manners)) + width * i, counts[i], width=width, label=age)

    ax.set_xticks(np.arange(len(manners)) + width * (len(ages) - 1) / 2)
    ax.set_xticklabels(manners, rotation=0, horizontalalignment='right', fontsize='16')
    ax.set_xlabel('Custody Status', fontsize='20')
    ax.set_ylabel('Count', fontsize='20')
    ax.legend(title='Age', title_fontsize='18')
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)
    plt.title('Counts of Custody Status by Age of Reduced Data', fontsize='16')
    plt.savefig('custody_status_by_age-reduced.png', bbox_inches='tight')
    plt.close()