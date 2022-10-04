# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 09:42:38 2021

@author: Administrateur
"""

#%% DNA classification using covid
import numpy as np
import os
path='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\npshortdataset'
dataarray=np.load(os.path.join(path,'Feature_labels_trainm.npy'))
dataarraym=np.load(os.path.join(path,'Feature_matrix_trainm.npy'))

label_1=dataarray[dataarray==1]
label_2=dataarray[dataarray==2]
label_3=dataarray[dataarray==3]
label_4=dataarray[dataarray==4]
label_5=dataarray[dataarray==5]

label1=np.zeros(len(label_1))
label2=np.ones(len(label_2))
label3=2*np.ones(len(label_3))
label4=3*np.ones(len(label_4))
label5=4*np.ones(len(label_5))
labeldata=[label1,label2,label3,label4,label5]
labelcovid=np.concatenate((label1,label2,label3,label4,label5), axis=0)
np.save('labeldatacovid.npy',labelcovid)