import os
import pandas as pd
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()
model = model.to(device)

ds_case = datasets.load_from_disk("/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/case_11000_ALLgenes")
ds_ctrl = datasets.load_from_disk("/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/ctrl_11000_ALLgenes")

gene_list = ['CRHR1','ESR1','ESR2','PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']
gene_count = len(gene_list)


save_dir = "/pscratch/sd/h/heehaw/GeneML/embeddings"
for gene in gene_list:
    save_dir_gene = f"{save_dir}/{gene}"
    if not os.path.isdir(save_dir_gene):
        os.mkdir(save_dir_gene)

for n in range(0, gene_count):

    print(gene_list[n]+' - start!')
    list_gene = list(range(n, gene_count*11000, gene_count))

    sublists = np.array_split(list_gene, 70)
    sublists = [sublist.tolist() for sublist in sublists]

    sublist_count = 1
    for lst in tqdm(sublists):
        #print('sublist '+str(sublist_count)+' - start!')
        sqs = []
        iids = []
        for i in lst:
            sq, iid, gene = islice(ds_case[i].values(), 3)
            iid_cat = iid + '_case'
            sqs.append(sq)
            iids.append(iid_cat)
        for i in lst:
            sq, iid, gene = islice(ds_ctrl[i].values(), 3)
            iid_cat = iid + '_ctrl'
            sqs.append(sq)
            iids.append(iid_cat)
        
        max_length = max(len(seq) for seq in sqs)
        #print(max_length)
        labels = [1 if '_case' in iid else 0 for iid in iids] # 1 for case, 0 for control
        np.save(f"{save_dir}/"+gene+'/labels_'+str(sublist_count)+'.npy', labels) # save labels
    
        # split the entire sequence into batches to avoid CUDA OOM error
        batch_size = 360
        batched_sqs = DataLoader(sqs, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=False)
    
        last_hidden_states = []
        batch_count = 1
        with torch.inference_mode():
            for batch in batched_sqs:
                inputs = tokenizer(batch, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt").to(device)
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
                
                del inputs, outputs, hidden_states, last_hidden_state, forward_hidden, rc_hidden, flipped_rc_hidden, averaged_hidden_state
                torch.cuda.empty_cache()
                #print('batch '+str(batch_count)+' - done!')
                batch_count += 1
        
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            last_hidden_state_cpu = last_hidden_states.cpu().numpy()
            embeddings = np.mean(last_hidden_state_cpu, axis=1) # mean pooling

        # save embeddings
        np.save(f"{save_dir}/"+gene+'/embeddings_'+str(sublist_count)+'.npy', embeddings)
    
        del last_hidden_states, last_hidden_state_cpu, embeddings
        #print('sublist '+str(sublist_count)+' - done!')
        sublist_count += 1
        torch.cuda.empty_cache()
        
    print(gene+' - done!!')
    print()
