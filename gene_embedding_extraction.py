import pandas as pd
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()
model = model.to(device)

ds_case = datasets.load_from_disk("dataset/case_50_ALLgenes")
ds_ctrl = datasets.load_from_disk("dataset/ctrl_50_ALLgenes")
gene_list = ['CRHR1','ESR1','ESR2','PCLO','FHIT','CACNA1C','DRD2','GRM7','EHD3','BICC1','PLOD1','LINC00687','CSMD1','LHPP','APC','ARHGAP8','LOC100996549','CNTNAP2','CRY1','COMT','FKBP5','HTR2A','BDNF','SLC6A4','ACE','SLC6A2','KCNK2','NR3C1','MTHFR','TPH1','TPH2','SOD2','CNR1','TNF','HTR1A','ABCB1','GNB3','GSK3B']
gene_count = len(gene_list)

for n in range(1, gene_count):

    print(gene_list[n]+' - start!')
    list_gene = list(range(n, gene_count*50, gene_count))

    sqs = []
    iids = []
    for i in list_gene:
        sq, iid, gene = islice(ds_case[i].values(), 3)
        iid_cat = iid + '_case'
        sqs.append(sq)
        iids.append(iid_cat)
    for i in list_gene:
        sq, iid, gene = islice(ds_ctrl[i].values(), 3)
        iid_cat = iid + '_ctrl'
        sqs.append(sq)
        iids.append(iid_cat)
    
    # batches to avoid CUDA OOM error
    batch_size = 20

    batched_sqs = DataLoader(sqs, batch_size=batch_size, shuffle=False)

    last_hidden_states = []
    with torch.no_grad(): # Disable gradient calculation for efficiency in inference
        for batch in batched_sqs:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            last_hidden_state = hidden_states[-1]
            last_hidden_states.append(last_hidden_state)
            torch.cuda.empty_cache()
            print('done')
        
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    print(last_hidden_states.shape)
    
    last_hidden_state_cpu = last_hidden_states.cpu().numpy()
    embeddings = np.mean(last_hidden_state_cpu, axis=1) # mean pooling
    print(embeddings)
    print(embeddings.shape)
    
    umap_model = umap.UMAP(n_components=2, random_state=98, n_neighbors=10)
    embedding_2d = umap_model.fit_transform(embeddings)
    print(embedding_2d.shape)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(embedding_2d[:,0], embedding_2d[:,1], s=50, c=categories, cmap='Spectral')
    plt.title('2D UMAP Projection of Embeddings ('+gene+')', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    handles, _ = scatter.legend_elements()
    plt.legend([handles[1], handles[0]], legend_labels)
    plt.savefig('visualization/gene_50-50/embedding_visualization_umap_'+gene+'.png')
    
    print(gene+' - done!')
    print()
