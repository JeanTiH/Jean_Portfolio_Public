""""""
"""OMSCS2024Spring-P4: Word Embeddings and Facial Recognition Analysis_Task1  	  		  		  		    	 		 		   		 		  

Student Name: Juejing Han 		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jhan446  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903845311		  	   		  	  		  		  		    	 		 		   		 		  
"""

import pandas as pd
import numpy as np
import gensim.models

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

newmodel = gensim.models.KeyedVectors.load_word2vec_format('/Users/jjhan/PycharmProjects/pythonProject1/AIES/P4/reducedvector.bin', binary=True)
a = newmodel.most_similar('man', topn=5)
b = newmodel.similarity('woman', 'man')
c = newmodel.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(a)
print(b)
print(c)
print('Example ends here!')
print('----------------------------')
#
print('----------------------------')
print('        Task set 1         ')
print('----------------------------')
print('***********')
print('    T1Q1   ')
print('***********')
targets = ['man', 'woman']
words = ['wife', 'husband', 'child', 'queen', 'king', 'man', 'woman', 'birth', 'doctor', 'nurse', 'teacher', 'professor', 'engineer','scientist', 'president']

simi_man = {}
simi_woman = {}
for target in targets:
    for word in words:
        simi = newmodel.similarity(target, word)
        if target == 'man':
            simi_man[word] = simi
        elif target == 'woman':
            simi_woman[word] = simi

# Sort the dictionary items by their values in descending order
sorted_simi_man = sorted(simi_man.items(), key=lambda x: x[1], reverse=True)
sorted_simi_woman = sorted(simi_woman.items(), key=lambda x: x[1], reverse=True)

df_man = pd.DataFrame(sorted_simi_man, columns=['List_man', 'Simi_man'])
df_woman = pd.DataFrame(sorted_simi_woman, columns=['List_woman', 'Simi_woman'])
df = pd.merge(df_man, df_woman, left_index=True, right_index=True)

df.to_csv('T1Q1.csv', index=False, float_format='%.3f')

print('***********')
print('    T1Q2   ')
print('***********')
filename = 'I01 [noun - plural_reg].txt'
column1 = 'Noun'
column2 = 'Plural'
# Read the text file into a DataFrame
df = pd.read_csv(filename, sep='\t', header=None, names=[column1, column2])
similarities = []
vocab_set = set(newmodel.index_to_key)

for index, row in df.iterrows():
    col1 = row[column1]
    col2 = row[column2]

    # Check if words are in the vocabulary
    if col1 in vocab_set and col2 in vocab_set:
        s0 = newmodel.similarity(col1, col2)
        similarity_score = "{:.3f}".format(s0)
    else:
        similarity_score = 'NA'

    similarities.append(similarity_score)

# Add similarities to the DataFrame
df['Similarity'] = similarities
# Compute similarity scores for the religion
words = ['christian', 'buddhist', 'muslim']

for word in words:
    col1_similarities = []

    for index, row in df.iterrows():
        col1 = row[column1]
        if col1 in vocab_set and word in vocab_set:
            similarity_score = newmodel.similarity(col1, word)
        else:
            similarity_score = 'NA'

        col1_similarities.append(similarity_score)

    df[word.capitalize() + '_Similarity'] = col1_similarities

df.to_csv('T1Q2.csv', index=False, float_format='%.3f')
print('***********')
print('    T1Q3   ')
print('***********')
print('--------')
print('   Q3a  ')
print('--------')
pairs = [ ['judge', 'bench'], ['genius', 'idiot'], ['jail', 'warden'], ['line', 'triangle'], ['dutch', 'netherlands'],
          ['king', 'queen'], ['liquid', 'solid'], ['sad', 'happy'], ['teacher', 'school'], ['japan', 'sushi'],
          ['dog', 'kennel'], ['sky', 'blue'], ['computer', 'cpu'], ['house', 'room'], ['sickness', 'health'] ]

similarities_human = []
for pair in pairs:
    word1, word2 = pair
    similarity_score = newmodel.similarity(word1, word2)
    similarities_human.append(similarity_score)

# Print the similarity scores
for pair, similarity_score in zip(pairs, similarities_human):
    print(f"Similarity score between '{pair[0]}' and '{pair[1]}': {similarity_score:.3f}")

print('--------')
print('   Q3b  ')
print('--------')
tripls = [ ['king', 'throne', 'judge'], ['giant', 'dwarf', 'genius'], ['college', 'dean', 'jail'],
          ['arc', 'circle', 'line'], ['french', 'france', 'dutch'], ['man', 'woman', 'king'],
          ['water', 'ice', 'liquid'], ['bad', 'good', 'sad'], ['nurse', 'hospital', 'teacher'],
          ['usa', 'pizza', 'japan'], ['human', 'house', 'dog'], ['grass', 'green', 'sky'],
          ['video', 'cassette', 'computer'], ['universe', 'planet', 'house'], ['poverty', 'wealth', 'sickness'] ]

similarities_model = []
for triple in tripls:
    model_output = newmodel.most_similar(positive=[triple[2], triple[1]], negative=[triple[0]], topn=1)
    similarities_model.append(model_output[0][1])
    print(f"Similarity score between '{triple[2]}' and '{model_output[0][0]}': {model_output[0][1]:.3f}")

print('--------')
print('   Q3c  ')
print('--------')
#print(similarities_human)
#print(similarities_model)
correlation = np.corrcoef(similarities_human, similarities_model)[0, 1]
print("Correlation coefficient:", correlation)