"""
Modified the original script to use DistributedDataParallel for faster inference, based on the following example:
https://github.com/huggingface/accelerate/blob/main/examples/inference/distributed/distributed_speech_generation.py

"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import pandas as pd
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import PartialState



def collate_fn(batch, distributed_state):
    # Extract input_ids from each sample in batch
    input_ids = torch.cat([sample['input_ids'] for sample in batch], dim=0)

    # Return dictionary with batched tensors
    return {
        'input_ids': input_ids.to(distributed_state.device),
    }



def create_dataloader(sqs, batch_size, distributed_state, tokenizer, max_length=100):
    """Create dataloader with preprocessing"""
    processed_dataset = [tokenizer(item, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt") for item in sqs]

    # Split dataset for distributed processing
    if distributed_state.num_processes > 1:
        chunk_size = len(processed_dataset) // distributed_state.num_processes
        start_idx = distributed_state.process_index * chunk_size
        end_idx = (
            start_idx + chunk_size
            if distributed_state.process_index < distributed_state.num_processes - 1
            else len(processed_dataset)
        )
        processed_dataset = processed_dataset[start_idx:end_idx]

    # Create batches
    batches = []
    for i in range(0, len(processed_dataset), batch_size):
        batch = processed_dataset[i : i + batch_size]
        batches.append(collate_fn(batch, distributed_state))
    return batches


def main():
    # Set Arguments 
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--control_data_dir", default="/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/ctrl_11000_ALLgenes", type=str, help="Data directory of Control Subjects")
    parser.add_argument("--case_data_dir", default="/global/cfs/projectdirs/m4244/heesun/NESAP/caduceus/dataset/case_11000_ALLgenes", type=str, help="Data directory of Case Subjects")
    parser.add_argument("--save_dir", default="/pscratch/sd/h/heehaw/GeneML/embeddings", type=str, help="Save directory")
    parser.add_argument("--batch_size", default=120, type=int, help="Batch size")
    args = parser.parse_args()


    # Set CUDA environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set Dataset and Gene List
    ds_case = datasets.load_from_disk(args.case_data_dir)
    ds_ctrl = datasets.load_from_disk(args.control_data_dir)

    gene_list = ['PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']
    gene_count = len(gene_list)


    # Set Save Directory
    for gene in gene_list:
        save_dir_gene = f"{args.save_dir}/{gene}"
        if not os.path.exists(save_dir_gene):
            os.mkdir(save_dir_gene)


    # Initialize Distributed State for Distributed Inference
    distributed_state = PartialState()

    # Initialize Model and Tokenizer
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True,device_map=distributed_state.device)
    model.eval()

    # Inference
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
            labels = [1 if '_case' in iid else 0 for iid in iids] # 1 for case, 0 for control
            np.save(os.path.join(*[args.save_dir, gene, f'labels_{sublist_count}.npy']), labels) # save labels
        
            # split the entire sequence into batches to avoid CUDA OOM error
            batched_sqs = create_dataloader(sqs=sqs, batch_size=args.batch_size, distributed_state=distributed_state, tokenizer=tokenizer,max_length=max_length)
        
            last_hidden_states = []
            batch_count = 1
            with torch.inference_mode():
                for batch in batched_sqs:
                    outputs = model(**batch, output_hidden_states=True)
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
                    
                    del batch, outputs, hidden_states, last_hidden_state, forward_hidden, rc_hidden, flipped_rc_hidden, averaged_hidden_state
                    torch.cuda.empty_cache()
                    #print('batch '+str(batch_count)+' - done!')
                    batch_count += 1
            
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
                last_hidden_state_cpu = last_hidden_states.cpu().numpy()
                embeddings = np.mean(last_hidden_state_cpu, axis=1) # mean pooling

            # save embeddings
            np.save(os.path.join(*[args.save_dir, gene, f'embeddings_{sublist_count}.npy']), embeddings)
        
            del last_hidden_states, last_hidden_state_cpu, embeddings
            #print('sublist '+str(sublist_count)+' - done!')
            sublist_count += 1
            torch.cuda.empty_cache()
            
        print(gene+' - done!!')



if __name__ == "__main__":
    main()