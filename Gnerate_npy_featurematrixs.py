# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 10:29:10 2021

@author: Administrateur
"""

#%% second sequence method
import pandas as pd
import os
import numpy as np
path='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset'
readcsvfile=os.path.join(path,'DNA_5classdata.csv')
datafile=pd.read_csv(readcsvfile)
#datafile.head()
classes=datafile.loc[:,'class']
# generate list of DNA sequences
sequences=datafile.loc[:,'sequence']
#print(sequences)
dataset={}
i=0

# loop through sequences and split into individual nucleotides
for seq in sequences:
    # split into nucleotides, remove tab characters
    nucleotides=list(seq)
    nucleotides=[x for x in seq if x!='\t']
    
    # append class assignment
    #nucleotides.append(classes[i])
    
    # add to dataset
    dataset[i]=(nucleotides)

    #increment i
    i+=1
    
print(dataset)
df = pd.DataFrame.from_dict(dataset, orient='index')
numerical_df = pd.get_dummies(df)
numerical_df.describe()
feature_matrix=np.array(numerical_df)
np.save('second_method_covid.npy',feature_matrix)
#%
#%%
#https://github.com/nageshsinghc4/DNA-Sequence-Machine-learning/blob/master/DNA_sequence_classification.py
# from Bio import SeqIO
# for sequence in SeqIO.parse('example.fa', "fasta"):
#     print(sequence.id)
#     print(sequence.seq)
#     print(len(sequence))

#Ordinal encoding DNA sequence data¶
# function to convert a DNA sequence string to a numpy array
# converts to lower case, changes any non 'acgt' characters to 'n'
import numpy as np
import re
def string_to_array(seq_string):
    seq_string = seq_string.lower()
    seq_string = re.sub('[^acgt]', 'z', seq_string)
    seq_string = np.array(list(seq_string))
    return seq_string

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))

# function to encode a DNA sequence string as an ordinal vector
# returns a numpy vector with a=0.25, c=0.50, g=0.75, t=1.00, n=0.00
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else, lets say z
    return float_encoded

# seq_test = 'TTCAGCCAGTG'
# dd=ordinal_encoder(string_to_array(seq_test))

import pandas as pd
import os
import numpy as np
path='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset'
readcsvfile=os.path.join(path,'DNA_5classdata.csv')
datafile=pd.read_csv(readcsvfile)
#datafile.head()
classes=datafile.loc[:,'class']
# generate list of DNA sequences
sequences=datafile.loc[:,'sequence']

first_seq=sequences[0]

dd1=ordinal_encoder(string_to_array(first_seq))

featMat=[]
for i in range(0,len(datafile)):
    seq_test=datafile['sequence'][i]
    print(seq_test)
    ss=string_to_array(seq_test)
    eef=ordinal_encoder(ss) # one-hot encoder values for one sequence of chararter.
    #ff=eef.flatten()
    #ff=ff[1:600]
    featMat.append(eef)


import numpy as np

combined = np.hstack(featMat)

# convert different length array lst into numpy array
#a = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
a=featMat
import numpy as np
b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))])
for i,j in enumerate(a):
    b[i][0:len(j)] = j
    
np.save('ordinalencoding_covidfeatures.npy',b)

#pathla='C:\\Users\\Administrateur\\Downloads\\labeldatacovid.npy'
#lab=np.load(pathla)
    
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
path='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset'
readcsvfile=os.path.join(path,'DNA_5classdata.csv')
datafile=pd.read_csv(readcsvfile)
datafile.head()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
def one_hot_encoder(seq_string):
    #label_encoder=label_encoder.fit(seq_string)
    #int_encoded = label_encoder.transform(seq_string).fit()
    #int_encoded = label_encoder.transform(seq_string)
    int_encoded=label_encoder.fit_transform(seq_string)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int)
    int_encoded = int_encoded.reshape(len(int_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded

import numpy as np
import re
def string_to_array(seq_string):
    seq_string = seq_string.lower()
    seq_string = re.sub('[^acgt]', 'z', seq_string)
    seq_string = np.array(list(seq_string))
    return seq_string
seq_test=datafile['sequence'][0]
#seq_test = 'GAATTCTCGAA'
ss=string_to_array(seq_test)
eef=one_hot_encoder(ss) # one-hot encoder values for one sequence of chararter.
ff=eef.flatten()
featMat=[]
for i in range(0,len(datafile)):
    seq_test=datafile['sequence'][i]
    #print(seq_test)
    ss=string_to_array(seq_test)
    eef=one_hot_encoder(ss) # one-hot encoder values for one sequence of chararter.
    ff=eef.flatten()
    #ff=ff[1:600]
    featMat.append(ff)
    
#featurematrix=np.array(featMat)    
#dd=np.array(featurematrix)

a=featMat
import numpy as np
b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))])
for i,j in enumerate(a):
    b[i][0:len(j)] = j
    
b=b[:,1:10000]
np.save('onehotencod_covidfeatures_r.npy',b)



#%% generate feature matrix based on  sequence and labels
import os
import pandas as pd
pathcsv='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\Valintinnewdataset\\COVID_DNA_2021dataset\\reduced_form'
root=pathcsv
pathc1=os.path.join(root,'Covidclass1.csv')
pathc2=os.path.join(root,'Dengueclass2.csv')
pathc3=os.path.join(root,'Hepatitisclass3.csv')
pathc4=os.path.join(root,'influenzaclass4.csv')
pathc5=os.path.join(root,'MRSClass5.csv')
### read csv file
c1data=pd.read_csv(pathc1)
c2data=pd.read_csv(pathc2)
c3data=pd.read_csv(pathc3)
c4data=pd.read_csv(pathc4)
c5data=pd.read_csv(pathc5)
datafile = pd.concat([c1data, c2data,c3data,c4data,c5data], ignore_index=True)
datafile.to_csv('DNA_5classdata.csv',index=False)


#%%
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
def getKmers(sequence, size=8):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
        
def generate_dataset(dfs,kmer_size,max_features,split=5):
    kmer_dfs=[]
    for cur_df in dfs:
        cur_df['words']=cur_df.apply(lambda x: getKmers(x['sequence'],size=kmer_size), axis=1)
        cur_df=cur_df.drop('sequence',axis=1)
        kmer_dfs.append(cur_df)
            
    all_data=pd.concat(kmer_dfs).reset_index(drop=True)
    print(len(all_data))
    perm=np.random.permutation(len(all_data)) #shuffle the data
        
    train_data=all_data
    train_kmers=[]
    for cur_kmer_list in train_data.words.values:
        train_kmers.extend(cur_kmer_list)
    vectorizer = CountVectorizer(max_features=max_features).fit(train_kmers) 
    #cur_transformed=vectorizer.transform(cur_data)
    print(all_data["class"].value_counts())
    
    
    X_train=[]
    Y_train=[]
            
    for cur_data, label in zip(train_data['words'],train_data['class']):
        cur_transformed=vectorizer.transform(cur_data)
        X_train.append(cur_transformed.toarray().sum(axis=0))
        Y_train.append(label)
            
    return X_train,Y_train
        
pathcsv='C:\\Users\\Administrateur'
readcsvfile=os.path.join(pathcsv,'DNA_5classdata.csv')
datafile=pd.read_csv(readcsvfile)
small_animals=datafile
X_train, Y_train=generate_dataset([small_animals],kmer_size=6,max_features=2500)
#%% split dataset into training and validation
from sklearn.model_selection import train_test_split

xtraindata=np.array(X_train)
label_data=np.array(Y_train)

X_train1, X_test, y_train, y_test = train_test_split(xtraindata, label_data, test_size=0.20, random_state=1)

# summarize first 5 rows
print(X_train[:5, :])   
np.save('Feature_matrix_trainm.npy',xtraindata)
np.save('Feature_labels_trainm.npy',label_data)