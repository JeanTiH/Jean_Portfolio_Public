""""""
"""OMSCS2024Spring-P3: Toxicity Analysis 	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14, 'axes.labelsize': 14})
import numpy as np

'''
#################################################
            Step1 Data preprocessing
#################################################
'''

def preprocess(filename):
    data_raw = pd.read_csv(filename, low_memory=False)
    print('Raw Data Shape    : ', data_raw.shape)
    data_raw.dropna(inplace=True)
    print('Dropnan Data Shape: ', data_raw.shape)

    # Filter rows where all values in columns 3 to 52 are "False"
    columns_to_check = data_raw.columns[2:]
    data_reduced = data_raw[~data_raw[columns_to_check].eq(False).all(axis=1)].copy()
    print('Reduced Data Shape: ', data_reduced.shape)

    # Function for calculating ranks
    def calculate_ranks (data_reduced, columns):
        ave_dict = {column: None for column in columns}
        num_dict = {column: None for column in columns}

        # Calculate average_toxicity of each subgroup
        for column in columns:
            true_data = data_reduced[data_reduced[column] == True]
            average_toxicity = true_data['TOXICITY'].mean()
            ave_dict[column] = average_toxicity

        # Sort ave_dict by average_toxicity in ascending order
        sorted_ave_dict = {k: v for k, v in sorted(ave_dict.items(), key=lambda item: item[1])}
        # Assign ranks to num_dict based on sorted values
        rank = 1
        for key in sorted_ave_dict:
            num_dict[key] = rank
            rank += 1
        return num_dict, ave_dict

    # Function for later use to assgin num to each protected class
    def assign_num(row):
        true_columns = [column for column in columns if row[column]]
        if len(true_columns) == 1:
            return num_dict[true_columns[0]]
        elif len(true_columns) > 1:
            return round(np.mean([num_dict[column] for column in true_columns]))
        else:
            return 0
    # -------------------------------------------
    # 1. Gender Identity (5 subgroups)
    # --------------------------------------------
    # male, female, nonbinary, transgender, trans
    # -------------------------------------------
    columns = ['male', 'female', 'nonbinary', 'transgender', 'trans']
    sub_dict = {'Gender Identity': columns}
    num_dict, ave_dict = calculate_ranks(data_reduced, columns)
    data_reduced['Gender Identity'] = data_reduced.apply(assign_num, axis=1)
    # -------------------------------------------
    # 2. Sexual Orientation (9 subgroups)
    # --------------------------------------------
    # heterosexual, straight, homosexual, gay, lesbian, bisexual, queer, lgbt, lgbtq
    # -------------------------------------------
    columns = ['heterosexual', 'straight', 'homosexual', 'gay', 'lesbian', 'bisexual', 'queer', 'lgbt', 'lgbtq']
    sub_dict['Sexual Orientation'] = columns
    num_dict, ave_dict1 = calculate_ranks(data_reduced, columns)
    data_reduced['Sexual Orientation'] = data_reduced.apply(assign_num, axis=1)
    ave_dict.update(ave_dict1)
    # -------------------------------------------
    # 3. Race (17 subgroups)
    # --------------------------------------------
    # white, american, canadian, european, black, african american, african, asian, chinese, japanese, indian, middle eastern, hispanic, latino, latina, latinx, mexican
    # -------------------------------------------
    columns = ['white', 'american', 'canadian', 'european', 'black', 'african american', 'african', 'asian', 'chinese',
               'japanese', 'indian', 'middle eastern', 'hispanic', 'latino', 'latina', 'latinx', 'mexican']
    sub_dict['Race'] = columns
    num_dict, ave_dict1 = calculate_ranks(data_reduced, columns)
    data_reduced['Race'] = data_reduced.apply(assign_num, axis=1)
    ave_dict.update(ave_dict1)
    # -------------------------------------------
    # 4. Religion (8 groups)
    # --------------------------------------------
    # christian, catholic, protestant, muslim, jewish, buddhist, sikh, taoist
    # -------------------------------------------
    columns = ['christian', 'catholic', 'protestant', 'muslim', 'jewish', 'buddhist', 'sikh', 'taoist']
    sub_dict['Religion'] = columns
    num_dict, ave_dict1 = calculate_ranks(data_reduced, columns)
    data_reduced['Religion'] = data_reduced.apply(assign_num, axis=1)
    ave_dict.update(ave_dict1)
    # -------------------------------------------
    # 5. Age (8 subgroups)
    # --------------------------------------------
    # young, younger, teenage, middle aged, millenial, old, older, elderly
    # -------------------------------------------
    columns = ['young', 'younger', 'teenage', 'middle aged', 'millenial', 'old', 'older', 'elderly']
    sub_dict['Age'] = columns
    num_dict, ave_dict1 = calculate_ranks(data_reduced, columns)
    data_reduced['Age'] = data_reduced.apply(assign_num, axis=1)
    ave_dict.update(ave_dict1)
    # -------------------------------------------
    # 6. Disability (3 subgroups)
    # --------------------------------------------
    # blind, deaf, paralyzed
    # -------------------------------------------
    columns = ['blind', 'deaf', 'paralyzed']
    sub_dict['Disability'] = columns
    num_dict, ave_dict1 = calculate_ranks(data_reduced, columns)
    data_reduced['Disability'] = data_reduced.apply(assign_num, axis=1)
    ave_dict.update(ave_dict1)
    return data_reduced, ave_dict, sub_dict

def step4(data):
    mean_toxicity = data['TOXICITY'].mean()
    std_toxicity = data['TOXICITY'].std()
    low = mean_toxicity - 1.96 * std_toxicity
    high = mean_toxicity + 1.96 * std_toxicity

    print('Mean TOXICITY: ', mean_toxicity)
    print('Std TOXICITY:', std_toxicity)
    print('Range of 95% TOXICITY around the mean: ', (low, high))

    # Calculate the margin of error (using the Z-score for a 95% confidence interval)
    z_score = 1.96  # Z-score for 95% confidence interval
    margin_of_error = z_score * (std_toxicity / np.sqrt(len(data['TOXICITY'])))

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean_toxicity - margin_of_error
    upper_bound = mean_toxicity + margin_of_error
    print('Margin Error: ', margin_of_error)
    print("95% Confidence Interval for TOXICITY:", (lower_bound, upper_bound))
    return mean_toxicity, std_toxicity

def step6(data, sub_dict, chosen_class):
    columns = sub_dict[chosen_class]

    mean_toxicity_dict = {}
    std_toxicity_dict = {}
    moe_toxicity_dict = {}
    lower_toxicity_dict = {}
    upper_toxicity_dict = {}
    for column in columns:
        # Filter the subgroup within the chosen_class is False
        non_zero_data = data[data[column] != False]
        print(column, non_zero_data.shape)

        mean_toxicity = non_zero_data['TOXICITY'].mean()
        std_toxicity = non_zero_data['TOXICITY'].std()

        # Calculate the margin of error (using the Z-score for a 95% confidence interval)
        z_score = 1.96  # Z-score for 95% confidence interval
        margin_of_error = z_score * (std_toxicity / np.sqrt(len(non_zero_data['TOXICITY'])))

        # Calculate the lower and upper bounds of the confidence interval
        lower_bound = mean_toxicity - margin_of_error
        upper_bound = mean_toxicity + margin_of_error

        mean_toxicity_dict[column] = mean_toxicity
        std_toxicity_dict[column] = std_toxicity
        moe_toxicity_dict[column] = margin_of_error
        lower_toxicity_dict[column] = lower_bound
        upper_toxicity_dict[column] = upper_bound

    for column, mean_toxicity in mean_toxicity_dict.items():
        print(f"Mean TOXICITY (when {column} is non-False): {mean_toxicity}")
        print(f"Standard Deviation of TOXICITY (when {column} is non-False): {std_toxicity_dict[column]}")
        print(f"Margin of Error of TOXICITY (when {column} is non-False): {moe_toxicity_dict[column]}")
        print(f"Lower Bound of TOXICITY (when {column} is non-False): {lower_toxicity_dict[column]}")
        print(f"Upper Bound of TOXICITY (when {column} is non-False): {upper_toxicity_dict[column]}")
    return mean_toxicity_dict, std_toxicity_dict

'''
#################################################
            Read Data and Run Experiments
#################################################
'''
if __name__ == "__main__":
    filename = 'toxity_per_attribute.csv'
    data, ave_dict, sub_dict = preprocess(filename)

    # -----------------------------------------------
    # Dict for step7 plot
    mean_toxi_dict = {}
    std_toxi_dict = {}
    # -----------------------------------------------

    print('--------------------------------------------------')
    print('                       Step 3                     ')
    print('--------------------------------------------------')
    print()

    # Correlation (exclude when assigned num is zero: all subgroups are False)
    correlation_dict = {}

    # sub_dict stores protected class names as key, and each subgroup name as value
    for pro_class in sub_dict:
        non_zero_mask = data[pro_class] != 0  # Create a mask for non-zero values
        correlation = data.loc[non_zero_mask, 'TOXICITY'].corr(data.loc[non_zero_mask, pro_class])
        correlation_dict[pro_class] = correlation
        # Print the correlation coefficients
    for pro_class, correlation in correlation_dict.items():
        print(f"Correlation between TOXICITY and {pro_class}: {correlation}")

    # Plot top three

    for pro_class, columns in sub_dict.items():
        values = [ave_dict[column] for column in columns]

        # Combine the lists into pairs, sort them based on the values in ascending order, then unzip
        sorted_pairs = sorted(zip(values, columns))
        sorted_values, sorted_columns = zip(*sorted_pairs)
        sorted_values = list(sorted_values)
        sorted_columns = list(sorted_columns)
        x_labels = range(1, len(values) + 1)
        #print("Sorted Values:", sorted_values)
        #print("Sorted Columns:", sorted_columns)

        plt.figure(figsize=(10, 6))
        plt.scatter(x_labels, sorted_values, color='skyblue', marker='o')
        plt.xlabel(pro_class, fontsize=20)
        plt.ylabel('Toxicity', fontsize=20)
        plt.title('Average Toxicity by ' + pro_class, fontsize=16)
        plt.xticks(range(1, len(x_labels) + 1), x_labels)
        #plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{pro_class}.png', bbox_inches='tight')
        plt.close()

    print('--------------------------------------------------')
    print('                     Step 4                       ')
    print('--------------------------------------------------')
    print()
    print('*************************')
    print('Original Reduced Data')
    print('*************************')
    mean_toxicity, std_toxicity = step4(data)

    # -----------------------------------------------
    # For step7 plot
    mean_toxi_dict['Population'] = mean_toxicity
    std_toxi_dict['Population'] = std_toxicity
    #-----------------------------------------------

    # Sample 10%
    print('*************************')
    print('Sample 10% Reduced Data')
    print('*************************')
    sampled_data1 = data.sample(frac=0.1, random_state=42)
    print('10%', sampled_data1.shape)
    mean_toxicity, std_toxicity = step4(sampled_data1)

    # Sample 60%
    print('*************************')
    print('Sample 60% Reduced Data')
    print('*************************')
    sampled_data2 = data.sample(frac=0.6, random_state=42)
    print('60%', sampled_data2.shape)
    mean_toxicity, std_toxicity= step4(sampled_data2)

    print('--------------------------------------------------')
    print('                      Step 5                      ')
    print('--------------------------------------------------')
    print()
    chosen_class = 'Disability'
    # Filter chosen_class is False
    filtered_data = data[data[chosen_class] != 0]

    #******************************************************
    # Show chosen class info
    #******************************************************
    print('Chose Class           : ', chosen_class)
    print('Chose Class Data Shape: ', filtered_data.shape)
    # Get subgroup info
    columns = sub_dict[chosen_class]
    for column in columns:
        # Filter the subgroup within the chosen_class is False
        non_zero_data = data[data[column] != False]
        print(column, ' records: ', len(non_zero_data))
    #******************************************************

    print('*************************')
    print('Original Reduced Data')
    print('*************************')
    mean_toxicity, std_toxicity = step4(filtered_data)

    # -----------------------------------------------
    # For step7 plot
    mean_toxi_dict[chosen_class] = mean_toxicity
    std_toxi_dict[chosen_class] = std_toxicity
    # -----------------------------------------------

    # Sample 10%
    print('*************************')
    print('Sample 10% Reduced Data')
    print('*************************')
    #sampled_data1 = filtered_data.sample(frac=0.1, random_state=42)
    #mean_toxicity, std_toxicity = step4(sampled_data1)
    filtered_data1 = sampled_data1[sampled_data1[chosen_class] != 0]
    mean_toxicity, std_toxicity = step4(filtered_data1)

    # Sample 60%
    print('*************************')
    print('Sample 60% Reduced Data')
    print('*************************')
    #sampled_data2 = filtered_data.sample(frac=0.6, random_state=42)
    #mean_toxicity, std_toxicity = step4(sampled_data2)
    filtered_data2 = sampled_data2[sampled_data2[chosen_class] != 0]
    mean_toxicity, std_toxicity = step4(filtered_data2)

    print('--------------------------------------------------')
    print('                       Step 6                     ')
    print('--------------------------------------------------')
    print()
    print('Chose Class Same as Step5: ', chosen_class)

    print('*************************')
    print('Original Reduced Data')
    print('*************************')
    mean_toxicity_dict, std_toxicity_dict = step6(data, sub_dict, chosen_class)

    # -----------------------------------------------
    # For step7 plot
    mean_toxi_dict.update(mean_toxicity_dict)
    std_toxi_dict.update(std_toxicity_dict)
    # -----------------------------------------------

    # Sample 10%
    print('*************************')
    print('Sample 10% Reduced Data')
    print('*************************')
    mean_toxicity_dict, std_toxicity_dict = step6(filtered_data1, sub_dict, chosen_class)

    # Sample 60%
    print('*************************')
    print('Sample 60% Reduced Data')
    print('*************************')
    mean_toxicity_dict, std_toxicity_dict = step6(filtered_data2, sub_dict, chosen_class)

    print('--------------------------------------------------')
    print('                       Step 7                     ')
    print('--------------------------------------------------')
    print()
    # Extract keys and values from the dictionaries
    mean_keys = list(mean_toxi_dict.keys())
    mean_values = list(mean_toxi_dict.values())

    std_keys = list(std_toxi_dict.keys())
    std_values = list(std_toxi_dict.values())

    # Create a figure and axes object
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot bar chart for mean values
    ax1.bar(mean_keys, mean_values, color='skyblue', label='Mean Toxicity')
    #ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Mean Toxicity', fontsize=20)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for standard deviation values
    ax2 = ax1.twinx()
    ax2.set_ylim(0.3, 0.37)
    ax2.plot(std_keys, std_values, color='orange', marker='o', label='STDV Toxicity')
    ax2.set_ylabel('STDV Toxicity', fontsize=20)
    ax2.legend(loc='upper right')

    # Title and grid
    plt.title('Mean and Standard Deviation of Toxicity', fontsize=16)
    plt.grid(True)

    # Show plot
    plt.tight_layout()
    plt.savefig('step7.png', bbox_inches='tight')
    plt.close()