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
import ankh
import torch
from torch.utils.data import Dataset



class SeqDataset(Dataset):
    def __init__(self, data_Dir=None):
        #self.seq_len = 512
        self.sequences = self.load_seq_from_json(data_Dir=data_Dir)


    def load_seq_from_json(self, data_Dir):
        sequences = []

        file_list = glob.glob(os.path.join(data_Dir, '*.json'))
        #print(file_list)
        for file in file_list:
            # load data
            with open(file, 'r') as f:
                subject_seqs = json.load(f)
            # parsing data into tuple types
            for seq_per_gene in subject_seqs['sequence']:
                eid  = subject_seqs['eid']
                gene_id = seq_per_gene['id']
                amino_seq = seq_per_gene['amino_seq']
                sequences.append((eid, gene_id, amino_seq))

                #num_segs = len(amino_seq) // self.seq_len + 1
                #for seg_id in range(num_segs):
                #    if seg_id < num_segs - 1:
                #        sequences.append((eid, gene_id, seg_id, amino_seq[seg_id*self.seq_len: (seg_id+1)*self.seq_len]))
                #    elif seg_id == num_segs - 1:
                #        sequences.append((eid, gene_id, seg_id, amino_seq[seg_id*self.seq_len: -1]))
                #    else:
                #        raise ValueError("There is some error in this code")
        #print(sequences[0])
        return sequences

    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]




def main_case():
    ## model inferencing
    chunk = 0
    data_dir = f'/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/case_amino/'
    save_dir = f"/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/case_emb/"
    fasta_dataset = SeqDataset(data_Dir=data_dir)


    model, tokenizer = ankh.load_base_model()
    model.eval()
    model.cuda()

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(fasta_dataset), start=1):
            eid, gene_id, amino_seq = sample
            #assert len(amino_seq) <= fasta_dataset.seq_len
            
            # extract embedding
            input_ids = tokenizer.batch_encode_plus(
                [list(amino_seq)],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = model(input_ids=input_ids["input_ids"].cuda())[0]

            # check whether directory exist 
            if os.path.isdir(os.path.join(save_dir, eid)) is False:
                os.mkdir(os.path.join(save_dir, eid))
            #if os.path.isdir(os.path.join(*[save_dir, eid, f'gene_{gene_id}'])) is False:
            #    os.mkdir(os.path.join(*[save_dir, eid, f'gene_{gene_id}']))

            # save subsequence of amino-translated gene sequence
            file_name = os.path.join(*[save_dir, eid, f'gene_{gene_id}.npy'])
            np.save(file_name, embedding.detach().cpu().numpy())
            #torch.save(embedding.detach().cpu(), file_name)

          
        
        
def main_ctrl():
    ## model inferencing
    chunk = 0
    data_dir = f'/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/control_amino/'
    save_dir = f"/pscratch/sd/h/heehaw/MyQuota/UKBiobank/embedding_visualization/control_emb/"
    fasta_dataset = SeqDataset(data_Dir=data_dir)


    model, tokenizer = ankh.load_base_model()
    model.eval()
    model.cuda()

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(fasta_dataset), start=1):
            eid, gene_id, amino_seq = sample
            #assert len(amino_seq) <= fasta_dataset.seq_len
            
            # extract embedding
            input_ids = tokenizer.batch_encode_plus(
                [list(amino_seq)],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = model(input_ids=input_ids["input_ids"].cuda())[0]

            # check whether directory exist 
            if os.path.isdir(os.path.join(save_dir, eid)) is False:
                os.mkdir(os.path.join(save_dir, eid))
            #if os.path.isdir(os.path.join(*[save_dir, eid, f'gene_{gene_id}'])) is False:
            #    os.mkdir(os.path.join(*[save_dir, eid, f'gene_{gene_id}']))

            # save subsequence of amino-translated gene sequence
            file_name = os.path.join(*[save_dir, eid, f'gene_{gene_id}.npy'])
            np.save(file_name, embedding.detach().cpu().numpy())
            #torch.save(embedding.detach().cpu(), file_name)

if __name__ == '__main__':
    #main_case()
    main_ctrl()
