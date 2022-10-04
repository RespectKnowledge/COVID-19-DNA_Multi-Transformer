# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 11:51:30 2021

@author: Administrateur
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pathcsv='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset'
readcsvfile=os.path.join(pathcsv,'sequences.csv')
datafile=pd.read_csv(readcsvfile)
datafile.head()

# from Bio import SeqIO

# for record in SeqIO.parse("D:\\DNA_covid2021\\sequences.fasta", "fasta"):
#     print(record.id)
#     print(repr(record.seq))
#     print(len(record))

#classes=datafile['Species']['Severe acute respiratory syndrome-related coronavirus']
class1=datafile.loc[datafile["Species"] == "Severe acute respiratory syndrome-related coronavirus"]
#class2=datafile.loc[datafile["Species"] == "Dengue"]
#df = datafile[datafile['Species'].str.startswith('Dengue')]
class2=datafile[datafile.Species.str.contains('Dengue',na=False)]
class3=datafile[datafile.Species.str.contains('Hepatitis',na=False)]
class4=datafile[datafile.Species.str.contains('Influenza',na=False)]
class5=datafile[datafile.Species.str.contains('coronavirus',na=False)]
#class2['Accession']
Cirratulidaeclass='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\Dengue_class'
from Bio import SeqIO
from Bio import Entrez    
Geneinfo1_Cirratulidae=[]
for i in class2['Accession']:
    Geneinfo1_Cirratulidae.append(i)
    Entrez.email = "engr.qayyum@example.com"  # Always tell NCBI who you are
    filename = str(i)+".fasta"
    if not os.path.isfile(filename):
        net_handle = Entrez.efetch(
            db="nucleotide", id=str(i), rettype="fasta", retmode="text"
            )
        out_handle = open(filename, "w")
        out_handle.write(os.path.join(Cirratulidaeclass,net_handle.read()))
        out_handle.close()
        net_handle.close()
        print("Saved:",i) 
        
## hepatitisclass        
Influenzaclass='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\Influenzaclass'
from Bio import SeqIO
from Bio import Entrez    
Geneinfo1_Cirratulidae=[]
for i in class4['Accession']:
    Geneinfo1_Cirratulidae.append(i)
    Entrez.email = "engr.qayyum@example.com"  # Always tell NCBI who you are
    filename = str(i)+".fasta"
    if not os.path.isfile(filename):
        net_handle = Entrez.efetch(
            db="nucleotide", id=str(i), rettype="fasta", retmode="text"
            )
        out_handle = open(filename, "w")
        out_handle.write(os.path.join(Influenzaclass,net_handle.read()))
        out_handle.close()
        net_handle.close()
        print("Saved:",i)
        
## Hepatitisclass      
Hepatitisclass='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\Hepatitisclass'
from Bio import SeqIO
from Bio import Entrez    
Geneinfo1_Cirratulidae=[]
for i in class3['Accession']:
    Geneinfo1_Cirratulidae.append(i)
    Entrez.email = "engr.qayyum@example.com"  # Always tell NCBI who you are
    filename = str(i)+".fasta"
    if not os.path.isfile(filename):
        net_handle = Entrez.efetch(
            db="nucleotide", id=str(i), rettype="fasta", retmode="text"
            )
        out_handle = open(filename, "w")
        out_handle.write(os.path.join(Hepatitisclass,net_handle.read()))
        out_handle.close()
        net_handle.close()
        print("Saved:",i)
 
class6=class5['Accession'][300000:-1]        
## Covidclass       
Covidclass='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\covid2'
from Bio import SeqIO
from Bio import Entrez    
Geneinfo1_Cirratulidae=[]
for i in class5['Accession'][300000:-1]:
    Geneinfo1_Cirratulidae.append(i)
    Entrez.email = "engr.qayyum@example.com"  # Always tell NCBI who you are
    filename = str(i)+".fasta"
    if not os.path.isfile(filename):
        net_handle = Entrez.efetch(
            db="nucleotide", id=str(i), rettype="fasta", retmode="text"
            )
        out_handle = open(filename, "w")
        out_handle.write(os.path.join(Covidclass,net_handle.read()))
        out_handle.close()
        net_handle.close()
        print("Saved:",i)
#%% MARS sequences accession numbers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
pathcsv='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset'
readcsvfile=os.path.join(pathcsv,'sequences_MARS.csv')
datafile=pd.read_csv(readcsvfile)
datafile.head()
class5=datafile[datafile.Species.str.contains('Middle East respiratory syndrome-related',na=False)]

MARSclass='C:\\Users\\Administrateur\\Desktop\\Desktopmaterial\\ENIB_work\\Valintinnewdataset\\COVID_DNA_2021dataset\\Accession_numbers\\MARSclass1'
from Bio import SeqIO
from Bio import Entrez    
Geneinfo1_Cirratulidae=[]
for i in class5['Accession']:
    Geneinfo1_Cirratulidae.append(i)
    Entrez.email = "engr.qayyum@example.com"  # Always tell NCBI who you are
    filename = str(i)+".fasta"
    if not os.path.isfile(filename):
        net_handle = Entrez.efetch(
            db="nucleotide", id=str(i), rettype="fasta", retmode="text"
            )
        out_handle = open(filename, "w")
        #out_handle1 = open(filename, "r")
        #print(out_handle1.read())
        out_handle.write(os.path.join(MARSclass,net_handle.read()))
        out_handle.close()
        net_handle.close()
        print("Saved:",i)