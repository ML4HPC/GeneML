import os
import pandas as pd
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # reduce memory fragmentation -> help PyTorch manage memory more efficiently

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()
model = model.to(device)

ds_case = datasets.load_from_disk("/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/mdd_case_11476_ALLgenes")
ds_ctrl = datasets.load_from_disk("/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/mdd_ctrl_11476_ALLgenes")

gene_list = ['CRHR1','ESR1','ESR2','PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']

gene_count = len(gene_list)
sample_size = 11476

# To keep the shuffling identical across genes
fixed_generator = torch.Generator()
    
for n in range(0, gene_count):

    list_gene = list(range(n, gene_count*sample_size, gene_count))

    sublists = np.array_split(list_gene, 70) # increase this value if facing OOM error
    sublists = [sublist.tolist() for sublist in sublists]
    
    sample_ids_list = []
    ages_list = []
    sexes_list = []
    labels_list = []
    
    sublist_count = 1
    for lst in sublists:
        samples = []
        for i in lst:
            sq, iid, gene, age, sex, label = islice(ds_case[i].values(), 6)
            samples.append((sq, iid, age, sex, label))
        for i in lst:
            sq, iid, gene, age, sex, label = islice(ds_ctrl[i].values(), 6)
            samples.append((sq, iid, age, sex, label))
        
        max_length = max(len(seq) for seq, _, _, _, _ in samples)

        # Split the entire sequence into batches to avoid CUDA OOM error
        batch_size = 10  # reduce this value as well if OOM error keeps occurring
        fixed_generator.manual_seed(98)
        batched_samples = DataLoader(samples, batch_size=batch_size, shuffle=True, generator=fixed_generator)
    
        last_hidden_states = []
        
        batch_count = 1
        with torch.inference_mode():
            for batch in batched_samples:
                sequences, sample_ids, ages, sexes, labels = batch
            
                inputs = tokenizer(list(sequences), padding='max_length', max_length=max_length, truncation=True, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                last_hidden_state = hidden_states[-1]
                
                # Split into forward and RC (reverse complementary) parts
                d_model = last_hidden_state.shape[-1] // 2  # 256
                forward_hidden = last_hidden_state[..., :d_model]  # First half
                rc_hidden = last_hidden_state[..., d_model:]       # Second half

                # Flip RC hidden state along sequence length (-2) and channel (-1)
                flipped_rc_hidden = torch.flip(rc_hidden, dims=(-2, -1))

                # Average forward and flipped RC hidden states
                averaged_hidden_state = (forward_hidden + flipped_rc_hidden) / 2

                # Append the processed hidden state
                last_hidden_states.append(averaged_hidden_state)
                
                # Aggregate metadata
                sample_ids_list.extend(list(sample_ids))
                ages_list.extend(list(ages))
                sexes_list.extend(list(sexes))
                labels_list.extend(list(labels))
                
                del sequences, sample_ids, ages, sexes, labels, inputs, outputs, hidden_states, last_hidden_state, forward_hidden, rc_hidden, flipped_rc_hidden, averaged_hidden_state
                torch.cuda.empty_cache()
                
                batch_count += 1
        
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            torch.cuda.empty_cache()
            last_hidden_state_cpu = last_hidden_states.cpu().numpy()
            
        # max pooling
        embeddings = np.max(last_hidden_state_cpu, axis=1)
            
        # save embeddings and metadata
        np.save(gene+'/embeddings_'+str(sublist_count)+'.npy', embeddings)
        
        del last_hidden_states, last_hidden_state_cpu, embeddings
        sublist_count += 1
    
    # Save metadata (only for the first processed gene - same for others)    
    if n == 0:
        np.save('iids.npy', sample_ids_list)
        np.save('age.npy', ages_list)
        np.save('sex.npy', sexes_list)
        np.save('labels.npy', labels_list)
