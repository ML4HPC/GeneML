# GeneML

- **sequence_converter_sample.py** is a python script for translating from DNA sequence to amino acid sequence
- **embedding_extractor_sample.py** is a python script for extracting embeddings using protein language model
- **embedding_extraction/caduceus_embedding_extraction.py** is a python script for extracting embeddings of DNA sequence of each gene using Caduceus
- **embedding_extraction/caduceus_embedding_extraction_DataParallel.py** is a python script for extracting embeddings of DNA sequence of each gene using Caduceus with Distributed Data Parallel
  ```
  # To run this script, using the following command line (This requires huggingfaces "accelerate" library)
  torchrun --nproc_per_node {num_gpus} caduceus_embedding_extraction_accelerate.py --save_dir /path/to/save/embeddings

  ```
- **classification/run_main.py** is a python script that...<br>
  1) loads gene embeddings, demographic information, and labels<br>
  2) aggregates gene embeddings with four different methods (PCA, max pooling, mean pooling, concatenation)<br>
  3) quick-tunes gradient boosting-based models (XGBoost, LightGBM, CatBoost) to select the best one among them<br>
  4) evaluates model performance of five classification algorithms (Random Forest, Logistic Regression, MLP, 1DCNN, selected gradient boosting model)<br>
  5) saves modeling results using five metrics (AUC, AUPRC, precision, recall, f1)
