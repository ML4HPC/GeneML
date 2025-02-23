# GeneML

- **sequence_converter_sample.py** is a python script for translating from DNA sequence to amino acid sequence
- **embedding_extractor_sample.py** is a python script for extracting embeddings using protein language model
- **caduceus_embedding_extraction.py** is a python script for extracting embeddings of DNA sequence of each gene using Caduceus
- **caduceus_embedding_extraction_DataParallel.py** is a python script for extracting embeddings of DNA sequence of each gene using Caduceus with Distributed Data Parallel
  ```
  # To run this script, using the following command line (This requires huggingfaces "accelerate" library)
  torchrun --nproc_per_node {num_gpus} caduceus_embedding_extraction_accelerate.py --save_dir /path/to/save/embeddings

  ```
- **gene_embeddings_classification_train_only.py** is a python script for simple binary classification only with the training set of exon embeddings for each gene and prediction performance check using 5-fold CV
- **classification_modeling_gene_demo.py** is a python script for simple binary classification with gene embeddings and demographic information separately and AUC calculation using stratified 10-fold CV
