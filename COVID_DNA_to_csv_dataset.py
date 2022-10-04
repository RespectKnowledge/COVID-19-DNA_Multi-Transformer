# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:44:39 2021
Abdul Qayyum
"""

#%% COVID DNA classification data conversion 

################### class 1 ##################
import os
from Bio import SeqIO
from Bio.Seq import Seq
pathc2='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\Dengue_class'
lstdir=os.listdir(pathc2)
lstdir1=os.path.join(pathc2,lstdir[0])
#se=list(SeqIO.parse('Downloads\\sequence (1).fasta', "fasta"))


for seq_record in SeqIO.parse(lstdir1, "fasta"):
        print(seq_record.id)
        print(str(seq_record.seq))

dataf={'sequence':[],
       'class':[]}
for i in lstdir:
    #print(i)
    path=os.path.join(pathc2,i)
    #print(path)
    for seq_record in SeqIO.parse(str(i), "fasta"):
        print(seq_record.id)
        print(repr(seq_record.seq))
        # print(str(seq_record.seq))
        # print(len(seq_record))
        for index, letter in enumerate(seq_record.seq):
            print("%i %s" % (index, letter))
            print(len(seq_record.seq))
            print(str(seq_record.seq))
            dataf['sequence'].append(str(seq_record.seq))
            dataf['class'].append(0)
import pandas as pd
datamyclass1=pd.DataFrame.from_dict(dataf)

###################### class 2
import os
from Bio import SeqIO
from Bio.Seq import Seq
pathc2='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\Dengue_class'
lstdir=os.listdir(pathc2)
dataf1={'sequence':[],
       'class':[]}
for i in lstdir:
    #print(i)
    for seq_record in SeqIO.parse(str(i), "fasta"):
        #print(seq_record.id)
        print(str(seq_record.seq))
        # print(str(seq_record.seq))
        # print(len(seq_record))
    # for index, letter in enumerate(seq_record.seq):
    #     print("%i %s" % (index, letter))
    #     print(len(seq_record.seq))
    #     print(str(seq_record.seq))
    dataf1['sequence'].append(str(seq_record.seq))
    dataf1['class'].append(1)
    
import pandas as pd
datamyclass2=pd.DataFrame.from_dict(dataf1)
    
############ class3
import os
from Bio import SeqIO
from Bio.Seq import Seq
pathc2='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\MARSclass1'
lstdir=os.listdir(pathc2)
dataf2={'sequence':[],
       'class':[]}
for i in lstdir:
    #print(i)
    for seq_record in SeqIO.parse(str(i), "fasta"):
        print(seq_record.id)
        #print(repr(seq_record.seq))
        #print(str(seq_record.seq))
        # print(len(seq_record))
    for index, letter in enumerate(seq_record.seq):
        print("%i %s" % (index, letter))
        print(len(seq_record.seq))
        print(str(seq_record.seq))
    dataf2['sequence'].append(str(seq_record.seq))
    dataf2['class'].append(2)
    
import pandas as pd
datamyclass3=pd.DataFrame.from_dict(dataf2)
      
#df_row = pd.concat([datamyclass1, datamyclass3,datamyclass2])
    
df_row_reindex = pd.concat([datamyclass1, datamyclass3,datamyclass2], ignore_index=True)

df_row_reindex.to_csv('DNA_Threeclass.csv',index=False)

#%%
import os
from Bio import SeqIO
from Bio.Seq import Seq
from io import StringIO
pathc2='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\MARSclass'
lstdir=os.listdir(pathc2)
dataf2={'sequence':[],
       'class':[]}
for i in lstdir:
    print(i)
    path=os.path.join(pathc2,i)
    records = SeqIO.parse(path, "fasta")
    out_handle = StringIO()
    SeqIO.write(records, out_handle, "fasta")
    fasta_data = out_handle.getvalue()
    print(str(fasta_data))
    dataf2['sequence'].append(fasta_data)
    dataf2['class'].append(5)
    
import pandas as pd
datamyclass3=pd.DataFrame.from_dict(dataf2)
      
#df_row = pd.concat([datamyclass1, datamyclass3,datamyclass2])
    
#df_row_reindex = pd.concat([datamyclass1, datamyclass3,datamyclass2], ignore_index=True)

#df_row_reindex.to_csv('DNA_Threeclass.csv',index=False)

    