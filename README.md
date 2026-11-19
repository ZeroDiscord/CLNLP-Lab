# Computational Linguistics & Natural Language Processing Laboratory

**Name:** Aantriksh Sood  
**SAP ID:** 500124259  
**Specialisation Batch:** AI/ML B1  
**Subject:** Computational Linguistics & Natural Language Processing  
**Faculty:** Dr. Sahinur Rahman Laskar

## Overview
This repository contains the laboratory experiments conducted for the Computational Linguistics and Natural Language Processing course. The experiments start with data analysis and text preprocessing, move on to statistical language representations (TF-IDF, embeddings, N-gram models), and end with neural and Transformer-based models for classification and multilingual generation.

## Laboratory Experiments

| Exp No. | Title | Description | Link |
| :---: | :--- | :--- | :---: |
| **1** | **Data Analysis** | Exploratory data analysis on employee records using Pandas, Matplotlib, and Seaborn to uncover demographic and salary distributions. | [View Details](Experiment1/README.md) |
| **2** | **Basic Text Preprocessing** | Implementation of core natural language preprocessing pipelines including text cleaning, tokenization, and stop word removal utilizing NLTK and spaCy. | [View Details](Experiment2/README.md) |
| **3** | **Stemming, Lemmatization and Regular Expression** | Implementation of rule-based stemming (Porter), vocabulary-based lemmatization (WordNet), and pattern extraction using Regular Expressions. | [View Details](Experiment3/README.md) |
| **4** | **Term Frequency and NER** | Implementation of Term-Frequency, TF-IDF, and Named Entity Recognition using both NLP toolkits and native Python from-scratch approaches. | [View Details](Experiment4/README.md) |
| **5** | **Subword Tokenization and POS Tagging** | Implementation of advanced tokenization using BPE and SentencePiece (pretrained and from scratch), followed by Part-of-Speech analysis. | [View Details](Experiment5/README.md) |
| **6** | **GloVe Embeddings and N-gram Language Model** | GloVe word embeddings trained from scratch (NumPy) and loaded pretrained (Gensim) to find words most similar to "king", plus a from-scratch N-gram language model with back-off for next-word prediction. | [View Details](Experiment6/README.md) |
| **7** | **Sentiment Analysis (LR vs LSTM vs Transformer vs BERT)** | Comparative sentiment classification on the IMDB 50K reviews dataset using TF-IDF Logistic Regression, a BiLSTM, a vanilla Transformer encoder and fine-tuned BERT, evaluated with Precision, Recall, F1 and Confusion Matrices. | [View Details](Experiment7/README.md) |
| **8** | **Multilingual Abstractive Summarization (mT5 vs mBART)** | Abstractive summarization of Hindi news articles from the XL-Sum dataset with mT5 and mBART, compared using multilingual ROUGE-1/2/L and mBERT-based BERTScore. | [View Details](Experiment8/README.md) |

## Structure
Each experiment directory contains:
*   A comprehensive Markdown documentation (`README.md`) detailing the procedure, implementation, and learning outcomes.
*   The primary Jupyter Notebook (`.ipynb`) containing the executable Python code and output visualizations.
*   Required dataset files (`.csv`, `.txt`, `.jsonl`) or accompanying documentations (`.docx`, `.pdf`).

> Large external resources are downloaded automatically by the notebooks rather than committed: the IMDB dataset (Exp 7), pretrained GloVe vectors (Exp 6), and Hugging Face models/datasets (Exp 7–8).
