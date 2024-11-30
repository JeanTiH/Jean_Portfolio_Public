""""""
"""OMSCS2024Spring-P4: Word Embeddings and Facial Recognition Analysis_Task2  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import glob
import os
import pandas as pd

print('----------------------------')
print('        Task set 1         ')
print('----------------------------')
# Directory containing the JPG files
directory = 'crop_part1'
# List all JPG files in the directory
files = glob.glob(os.path.join(directory, '*.jpg'))
print('Number of Original Images:', len(files))

numbers = [os.path.basename(file).split('_')[:3] for file in files]
# Create DataFrame
df = pd.DataFrame(numbers, columns=['age', 'gender', 'race'])

# Convert columns to numeric, NaN value appears
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
# Drop NaN, convert all values to int
df.dropna(inplace=True)
print('Number of Images After Data Cleaning:', df.shape[0])
df['age'] = df['age'].astype(int)
df['gender'] = df['gender'].astype(int)
df['race'] = df['race'].astype(int)

# Output usage
N = 5   # Number of subgroups of age
age_output = ['0 to 20', '21 to 40', '41 to 60', '61 to 80', '81 to 116']
df_gender = pd.DataFrame(columns=age_output)
df_race = pd.DataFrame(columns=age_output)
gendr_output = ['male', 'female']
race_output = ['White', 'Black', 'Asian', 'Indian', 'Others']
# Age groups
df_age = [None] * N
df_age[0] = df[(df['age'] >= 0) & (df['age'] <= 20)]
df_age[1] = df[(df['age'] >= 21) & (df['age'] <= 40)]
df_age[2] = df[(df['age'] >= 41) & (df['age'] <= 60)]
df_age[3] = df[(df['age'] >= 61) & (df['age'] <= 80)]
df_age[4] = df[(df['age'] >= 81) & (df['age'] <= 116)]

# Gender
for j in range(2):
    for i in range(N):
        df_sub = df_age[i][df_age[i]['gender'] == j]
        frequency = len(df_sub)
        #print('Frequency when age is from', age_output[i], 'and gender is', gendr_output[j] + ':', frequency)
        df_gender.loc[j, age_output[i]] = frequency
df_gender.to_csv('T2_gender.csv', index=False)
# Race
for j in range(5):
    for i in range(N):
        df_sub = df_age[i][df_age[i]['race'] == j]
        frequency = len(df_sub)
        #print('Frequency when age is from', age_output[i], 'and race is', race_output[j] + ':', frequency)
        df_race.loc[j, age_output[i]] = frequency
df_race.to_csv('T2_race.csv', index=False)