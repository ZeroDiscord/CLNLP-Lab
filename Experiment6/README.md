# Experiment 6: GloVe Word Embeddings and N-gram Language Model

**Name:** Aantriksh Sood  
**SAP ID:** 500124259  

## Objective
To implement the GloVe (Global Vectors for Word Representation) algorithm to generate word embeddings from a text corpus and find the words most similar to **"King"**, using two approaches (from scratch, and a pretrained Gensim model). The experiment also builds a statistical N-gram language model from scratch that predicts the next word in a sequence from the preceding words.

## Procedure & Implementation
This experiment covers two classical ways of modelling language statistically: dense distributional embeddings and count-based sequence probabilities.

1.  **GloVe Without a Pretrained Model (`6.1 a`):**
    *   Preprocessing the provided corpus (`glove_corpus.txt`, 24 sentences): lowercasing, alphabetic tokenisation and removal of high-frequency function words <sup>(`re.findall()`, stop-word `set`)</sup>.
    *   Building a symmetric, distance-weighted co-occurrence matrix $X_{ij}$ with a context window of 2 <sup>(`np.zeros((V, V))`, `1/abs(i-j)`)</sup>.
    *   Optimising the GloVe weighted least-squares objective $f(X_{ij})(w_i^\top\tilde{w}_j+b_i+\tilde{b}_j-\log X_{ij})^2$ with AdaGrad, using vectorised NumPy gradients <sup>(`np.add.at()`, `x_max=10`, `α=0.75`)</sup>.
    *   Using $w_i+\tilde{w}_i$ as the final embedding and ranking neighbours by cosine similarity <sup>(`np.linalg.norm()`, `np.argsort()`)</sup>.
    *   Plotting the loss curve and a 2-D PCA projection of the embedding space <sup>(`np.linalg.svd()`)</sup>.

2.  **GloVe Using a Pretrained Model (`6.1 b`):**
    *   Loading 100-d GloVe vectors trained on Wikipedia + Gigaword (6B tokens, 400K words) <sup>(`gensim.downloader.load('glove-wiki-gigaword-100')`)</sup>.
    *   Retrieving the nearest neighbours of "king" over the full vocabulary <sup>(`most_similar()`)</sup>, and again restricted to the corpus vocabulary for a direct comparison with the scratch model <sup>(`similarity()`)</sup>.
    *   Testing linear semantic structure with the analogy *king − man + woman* <sup>(`most_similar(positive=, negative=)`)</sup>.

3.  **N-gram Language Model From Scratch (`6.2`):**
    *   Segmenting the given corpus into 8 sentences and padding them with `<s>` / `</s>` boundary tokens.
    *   Implementing an `NGramLanguageModel` class that stores counts for every order up to $N$ <sup>(`defaultdict(int)`)</sup> and estimates $P(w_n \mid w_{n-N+1}^{n-1})$ by MLE with optional add-$k$ smoothing.
    *   Predicting the top-3 next words for several contexts with bigram, trigram and 4-gram models, **backing off** to a shorter context when a context is unseen <sup>(`predict_next()`)</sup>.
    *   Generating text greedily from seed words <sup>(`generate()`)</sup> and evaluating models with perplexity on training and held-out sentences <sup>(`perplexity()`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Co-occurrence Statistics** | Sparsity of the 60 × 60 matrix (188 non-zero pairs, 5.2%) | Distance-weighted window counting |
| **Scratch GloVe Training** | Weighted loss convergence (2.21 → ≈ 0) | Loss curve (log scale) & PCA scatter |
| **Similar Words (Scratch)** | Top-5 cosine neighbours of "king" | Custom cosine-similarity ranking |
| **Similar Words (Pretrained)** | Top-10 neighbours & king − man + woman analogy | Gensim `KeyedVectors` |
| **N-gram Prediction** | Next-word probability distribution per context | Bigram / Trigram / 4-gram with back-off |
| **Model Evaluation** | Train vs. held-out perplexity across $N$ | Perplexity comparison table |

### Key Results
| Task | Result |
| :--- | :--- |
| Most similar to "king" (scratch) | **queen** (0.970), son (0.904), powerful (0.725), wise, teaches |
| Most similar to "king" (pretrained) | **prince** (0.768), queen (0.751), son, brother, monarch, throne, kingdom |
| Analogy king − man + woman | **queen** (0.770) |
| Next word after *"natural language"* | Bigram: processing 0.43 / models 0.29 · Trigram: **processing 1.00** |
| Perplexity (train / held-out) | 2-gram 1.80 / **16.66** · 3-gram 1.44 / 21.21 · 4-gram 1.28 / 23.79 |

## Result Interpretation & Learnings
*   **Global Co-occurrence Encodes Meaning:** Even on a tiny corpus, GloVe places *king* next to *queen* because the two words share identical contexts (*rules … kingdom*, *lives … royal palace*). This supports the distributional hypothesis behind word embeddings.
*   **Data Scale Matters:** The scratch model drives its loss to ≈ 0 and memorises 188 co-occurrence pairs, so neighbours beyond the top 3 are noise. The pretrained vectors produce a clean royalty/kinship cluster and support vector arithmetic (*king − man + woman ≈ queen*), a linear structure that only appears with billions of tokens.
*   **Context Length vs. Sparsity:** Longer N-gram histories give sharper predictions (trigram certainty versus bigram ambiguity after *"natural language"*). They also fragment the counts: training perplexity falls as $N$ grows while held-out perplexity rises. This is the bias-variance trade-off of count-based language models.
*   **Smoothing & Back-off Are Essential:** Unseen contexts and words would otherwise get zero probability (infinite perplexity). Add-$k$ smoothing and back-off to lower orders keep the model usable on unseen text, and these limits are what motivated neural language models.
