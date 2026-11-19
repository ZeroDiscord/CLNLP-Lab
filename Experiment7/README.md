# Experiment 7: Sentiment Analysis on IMDB Movie Reviews (Logistic Regression, LSTM, Transformer & BERT)

**Name:** Aantriksh Sood  
**SAP ID:** 500124259  

## Objective
To implement binary sentiment analysis on the IMDB 50K Movie Reviews dataset using four model families—**Logistic Regression**, **Bidirectional LSTM**, **Vanilla Transformer Encoder**, and pretrained **BERT**—and to perform a comparative analysis across all models using **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices**.

## Procedure & Implementation
All four architectures are evaluated on the exact same stratified split of the deduplicated IMDB dataset (`35,698` train / `3,967` validation / `9,917` test reviews) to ensure a fair, controlled comparison.

1.  **Dataset Preprocessing & Stratified Splitting (`7.1`):**
    *   Loading the 50,000 IMDB movie reviews and removing `418` duplicate reviews to prevent train-test data leakage <sup>(`pd.read_csv()`, `drop_duplicates()`)</sup>.
    *   Cleaning HTML tags (`<br />`), normalizing whitespace, and creating both a lowercase alphabetic representation for from-scratch models and a punctuation-preserved representation for BERT <sup>(`re.sub()`)</sup>.
    *   Partitioning the corpus into stratified training, validation, and held-out test sets <sup>(`train_test_split(stratify=...)`)</sup>.

2.  **Logistic Regression with TF-IDF (`7.2`):**
    *   Extracting sparse unigram and bigram TF-IDF features (`max_features=100,000`, `sublinear_tf=True`) so local negations (e.g., *"not good"*) are captured <sup>(`TfidfVectorizer(ngram_range=(1, 2))`)</sup>.
    *   Training an L2-regularized Logistic Regression classifier (`C=4.0`) and inspecting the highest-magnitude positive and negative feature weights <sup>(`LogisticRegression()`, `coef_`)</sup>.

3.  **Bidirectional LSTM From Scratch (`7.3–7.4`):**
    *   Building a `20,000`-word vocabulary from the training split and encoding sequences up to `256` tokens <sup>(`Counter.most_common()`, `TensorDataset`)</sup>.
    *   Implementing a PyTorch `LSTMClassifier` with learned `128`-d embeddings, a packed bidirectional LSTM (`hidden_size=128`), dropout (`0.4`), and a linear classification head on the concatenated forward and backward final states <sup>(`nn.Embedding`, `pack_padded_sequence()`, `nn.LSTM(bidirectional=True)`)</sup>.
    *   Training with `BCEWithLogitsLoss`, Adam (`lr=1e-3`), gradient clipping, and validation-F1 checkpointing.

4.  **Vanilla Transformer Encoder From Scratch (`7.5`):**
    *   Implementing sinusoidal `PositionalEncoding` ($\sin/\cos$ over geometric frequencies) and scaling token embeddings by $\sqrt{d_{\text{model}}}$ <sup>(`register_buffer()`)</sup>.
    *   Stacking `2` multi-head self-attention encoder layers (`d_model=128`, `nhead=4`, `dim_feedforward=256`) with padding masks and masked mean pooling across valid tokens <sup>(`nn.TransformerEncoderLayer`, `nn.TransformerEncoder`)</sup>.

5.  **Pretrained BERT Fine-Tuning (`7.6`):**
    *   Tokenizing reviews into WordPiece subwords using `bert-base-uncased` (`max_length=256`) <sup>(`BertTokenizerFast.from_pretrained()`)</sup>.
    *   Fine-tuning `BertForSequenceClassification` (`109.48M` parameters) end-to-end with AdamW (`lr=2e-5`, `weight_decay=0.01`), linear warmup + decay schedule, and `bfloat16` mixed precision <sup>(`BertForSequenceClassification.from_pretrained()`, `get_linear_schedule_with_warmup()`, `torch.autocast()`)</sup>.

6.  **Comparative Evaluation (`7.7`):**
    *   Computing **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices** (with row-normalized rates and False Positive / False Negative breakdowns) on the held-out `9,917` test reviews <sup>(`precision_score()`, `recall_score()`, `f1_score()`, `confusion_matrix()`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Corpus Exploration** | Class balance & review length distribution (`256`-word cutoff) | Bar chart & histogram |
| **Linear Baseline (TF-IDF + LR)** | High-dimensional sparse n-gram weights & rapid convergence | Top positive/negative n-grams & classification report |
| **Recurrent Model (BiLSTM)** | Sequential gating over packed variable-length sequences | Validation F1 progression across epochs |
| **Self-Attention (Transformer)** | Non-pretrained global self-attention with masked mean pooling | Validation F1 progression across epochs |
| **Transfer Learning (BERT)** | Contextual subword representations via pretrained bidirectional encoder | Fine-tuning loss & validation F1 |
| **Cross-Model Comparison** | Precision, Recall, F1, Accuracy & Confusion Matrix error profiles | Heatmap grid & grouped metric bar chart |

### Comparative Results (Test Set: 9,917 Reviews)
| Model | Accuracy | Precision | Recall | F1-Score | False Positives | False Negatives | Train Time (s) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (TF-IDF 1–2 grams)** | 0.9117 | 0.9036 | 0.9224 | 0.9129 | 490 | 386 | 32.0s | 0.10M |
| **BiLSTM (From Scratch)** | 0.8748 | 0.8746 | 0.8760 | 0.8753 | 625 | 617 | 106.3s | 2.82M |
| **Vanilla Transformer (From Scratch)** | 0.8280 | 0.8306 | 0.8256 | 0.8281 | 838 | 868 | 87.8s | 2.83M |
| **BERT (`bert-base-uncased`, Fine-Tuned)** | **0.9190** | **0.9147** | **0.9249** | **0.9198** | **429** | **374** | 487.6s | 109.48M |

## Result Interpretation & Learnings
*   **Strong Linear Baseline for Bag-of-N-grams:** TF-IDF + Logistic Regression achieves competitive accuracy in a few seconds because sentiment in movie reviews is heavily signaled by explicit lexical phrases and bigrams (*"waste of time"*, *"highly recommend"*), though it cannot resolve long-range compositional syntax or sarcasm.
*   **Inductive Bias When Training From Scratch:** On a medium-sized corpus (`~35.7k` reviews), a BiLSTM trained from scratch outperforms a vanilla Transformer encoder trained from scratch. Recurrence provides a strong sequential inductive bias, whereas self-attention lacks built-in locality and typically requires large-scale pretraining to learn syntactic structure effectively.
*   **Transfer Learning Advantage (BERT):** Fine-tuning `bert-base-uncased` achieves the highest Precision, Recall, and F1-score, substantially reducing both False Positives and False Negatives. Pretraining on billions of tokens equips BERT with deep contextual semantics and subword coverage before seeing a single task label.
*   **Accuracy vs. Compute Trade-off:** While BERT delivers the best classification metrics, TF-IDF + Logistic Regression trains two orders of magnitude faster on CPU with `<0.2%` of the parameters, illustrating the practical deployment trade-off between latency and accuracy.
