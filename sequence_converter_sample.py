import os
import glob
import numpy as np
import pickle, sys
import pandas as pd
from Bio import SeqIO
import glob 
import re 
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import json

def IO_single_fasta_file(file_dir, return_amino_seq=False) -> list:
    subject_index = SeqIO.index(file_dir, "fasta")
    subject_all = []
    #print(len(subject_index))
    for i in range(len(subject_index)): 
        #print(i)
        subject_data = {}
        sequence = subject_index[f"{i+1}"]
        subject_data['id'] = sequence.id 
        gene_seq = str(sequence.seq)
        subject_data['seq'] = gene_seq
        subject_data['description'] = sequence.description
        if return_amino_seq == True: 
            subject_data['amino_seq'] = translate(gene_seq)
        else: 
            subject_data['amino_seq'] = None 
        subject_all.append(subject_data)
    return subject_all 


def check_nan_nucleotide(seq, replace='remove'):
    if "N" in seq: 
        if replace=='remove':
            seq = seq.replace("N", "")
    return seq
    

def translate(seq):
    """
    ref: https://www.geeksforgeeks.org/dna-protein-python-3/
    
    **stop codons (i.e., 'TAA', 'TAG', 'TGA') are removed from the lookup table since alphafold only support 20 amnio acids.
    """
    # first check if 'N' characters are included in the sequence 
    seq = check_nan_nucleotide(seq)

    
    table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    protein =""
    for i in range(0, len(seq)-2, 3):
        #print(i)
        codon = seq[i:i + 3]
        protein+= table[codon]
    # removing stop codon
    protein = protein.replace('_', '')
    return protein


def process_file(file_dir, return_amino_seq):
    subject_data = {}
    eid = os.path.split(file_dir)[-1].split('_')[-2]
    subject_data['eid'] = eid
    subject_data['sequence'] = IO_single_fasta_file(file_dir, return_amino_seq)    
    return subject_data


def data_preprocess(data_dir, save_dir):    
    ## since every subjects have both .fasta file and .fasta.fai file, we only use .fasta file format
    #fasta_case = glob.glob(os.path.join(case_dir, "*.fasta"))[:2]
    #fasta_ctrl = glob.glob(os.path.join(ctrl_dir, "*.fasta"))[:2]
    fasta_data = glob.glob(os.path.join(data_dir, "*.fasta"))
    
    ## multi processing
    cpu_cores = os.cpu_count()
    pool = Pool(processes=cpu_cores//16)  # setting the number of processes
    
    return_amino_seq = True
    
    # preprocess data 
    all_data = []
    for result in tqdm(pool.imap_unordered(partial(process_file,return_amino_seq=return_amino_seq), fasta_data), total=len(fasta_data)):
        all_data.append(result)
        file_dir = os.path.join(save_dir, f'{result["eid"]}.json')
        with open(file_dir, 'w') as file: 
            json.dump(result, file)
        #print(f'Subject Id: {result["eid"]}. Done')
        
    pool.close()
    pool.join()
    
    return all_data


def main(): 
    chunk = 0
    ## data preprocessing
    case_dir = f'/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/case'
    save_case_dir = f'/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/case_amino'
    case_all_data = data_preprocess(case_dir, save_case_dir)
    
    ctrl_dir = '/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/control'
    save_ctrl_dir = '/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/control_amino'
    ctrl_all_data = data_preprocess(ctrl_dir, save_ctrl_dir)


if __name__ == "__main__": 
    main()    
