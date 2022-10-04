# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:18:23 2021

@author: Administrateur
"""
#%% first dataset generator
import torch
from torch.utils.data import Dataset,DataLoader
class DNA_dataloader(Dataset):
    def __init__(self,root,size):
        super().__init__()
        self.root=root
        #self.csv_file=csv_file
        self.size=size
        readcsvfile=os.path.join(self.root,'DNA_5classdata.csv')
        datafile=pd.read_csv(readcsvfile)
        datafile.head()
        # function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
        def getKmers(sequence, size=6):
            return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

        # Now we can convert our training data sequences into short overlapping k-mers of legth 6. 
        # Lets do that for each species of data we have using our getKmers function.

        datafile['words'] = datafile.apply(lambda x: getKmers(x['sequence']), axis=1)
        datafile = datafile.drop('sequence', axis=1)
        datafile.head()

        data_texts = list(datafile['words'])
        for item in range(len(data_texts)):
            data_texts[item] = ' '.join(data_texts[item])

        self.y_data = datafile.iloc[:, 0].values

        print(data_texts[2])

        #Now we will apply the BAG of WORDS using CountVectorizer using NLP
        # Creating the Bag of Words model using CountVectorizer()
        # This is equivalent to k-mer counting
        # The n-gram size of 4 was previously determined by testing
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(ngram_range=(4,4))
        self.X = cv.fit_transform(data_texts).toarray()
        #print(X.shape)
        
    def __getitem__(self, index):
        
        X1=self.X[index,0:6000]  # X=number of samplex features
        #X1=X1[:,0:6000]
        Xd= np.reshape(X1, (20,300)) # reshape into(timestepsxnumber of features)
        y_data1=self.y_data[index] # y_data=labels
        
        return Xd,y_data1
    
    def __len__(self):
        
        return (len(self.X))
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pathcsv='C:\\Users\\Administrateur'
dataset=DNA_dataloader(root=pathcsv,size=6)
#%%
n_train_examples = int(len(dataset)*0.80)
n_valid_examples = len(dataset) - n_train_examples

train_set, val_set = torch.utils.data.random_split(dataset, 
                                                       [n_train_examples, n_valid_examples])

train_loader=DataLoader(train_set,batch_size=16,shuffle=True)
valid_loader=DataLoader(val_set,batch_size=4,shuffle=False) 
len(train_loader.dataset)
#len(valid_loader.dataset)
#batch,la=next(iter(train_loader))
for i,data in enumerate(train_loader):
    batch,labl=data
    print(batch.shape)
    print(labl)
    
for i,data in enumerate(valid_loader):
    batch,labl=data
    print(batch.shape)
    print(labl)

#%% third dataset generator

import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
import pickle
import matplotlib.pyplot as plt
class DNA_dataloader_3(Dataset):
    def __init__(self,root,ksize,max_f):
        super().__init__()
        self.root=root
        self.ksize=ksize
        self.max_f=max_f
        #self.csv_file=csv_file
        #self.size=size
        readcsvfile=os.path.join(self.root,'DNA_5classdata.csv')
        datafile=pd.read_csv(readcsvfile)
    
        small_animals=datafile
        
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
        
        X_train, Y_train=generate_dataset([small_animals],kmer_size=self.ksize,max_features=self.max_f)
        
        self.xtraindata=np.array(X_train)
        self.label_data=np.array(Y_train)
        
    def __getitem__(self, index):
        
        X1=self.xtraindata[index,:]  # X=number of samplex features
        #Xd=X1
        Xd= np.expand_dims(X1,axis=0) # (batchxseq_lenxfeatures)expand diemsion for LSTM and 1DCNN models
        labels=self.label_data
        y_label=labels[index]
        
        return Xd,y_label
    
    def __len__(self):
        
        return (len(self.xtraindata))

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pathcsv='C:\\Users\\Administrateur'
dataset=DNA_dataloader_3(root=pathcsv,ksize=6,max_f=2500)
#%% third dataset generator
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
import pickle
import matplotlib.pyplot as plt
class DNA_dataloader_3(Dataset):
    def __init__(self,X_data,y_data):
        super().__init__()
        self.X_data=X_data
        self.y_data=y_data
        X_train=np.load(self.X_data)
        Y_train=np.load(self.y_data)
        self.xtraindata=X_train
        self.label_data=Y_train
        
    def __getitem__(self, index):
        
        X1=self.xtraindata[index,:]  # X=number of samplex features
        #Xd=X1
        Xd= np.expand_dims(X1,axis=0) # (batchxseq_lenxfeatures)expand diemsion for LSTM and 1DCNN models
        labels=self.label_data
        y_label=labels[index]
        
        return Xd,y_label
    
    def __len__(self):
        
        return (len(self.xtraindata))
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
X_datap='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\npshortdataset\\Feature_matrix_trainm.npy'
y_data='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\npshortdataset\\Feature_labels_trainm.npy'
dataset=DNA_dataloader_3(X_data=X_datap,y_data=y_data)
##################### check dataset and dataloader class
n_train_examples = int(len(dataset)*0.80)
n_valid_examples = len(dataset) - n_train_examples

train_set, val_set = torch.utils.data.random_split(dataset, 
                                                       [n_train_examples, n_valid_examples])

train_loader=DataLoader(train_set,batch_size=16,shuffle=True)
valid_loader=DataLoader(val_set,batch_size=16,shuffle=False) 
len(train_loader.dataset)
len(valid_loader.dataset)
#batch,la=next(iter(train_loader))
for i,data in enumerate(train_loader):
    batch,labl=data
    print(batch.shape)
    print(labl)
    
for i,data in enumerate(valid_loader):
    batch,labl=data
    print(batch.shape)
    print(labl)
    
#%%
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
###### savining feature matrix for training and testing
#np.save('Feature_matrix_train.npy',X_train1)
#np.save('Feature_labels_train.npy',y_train)

#np.save('Feature_matrix_test.npy',X_test)
#np.save('Feature_labels_test.npy',y_test)


