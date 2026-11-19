# Experiment 8: Multilingual Abstractive Text Summarization (mT5 vs mBART on XL-Sum Hindi)

**Name:** Aantriksh Sood  
**SAP ID:** 500124259  

## Objective
To implement and evaluate abstractive text summarization using two multilingual Transformer sequence-to-sequence architectures—**mT5** (*Multilingual Text-to-Text Transfer Transformer*) and **mBART** (*Multilingual Bidirectional and Auto-Regressive Transformers*)—on the **Hindi (`hi`)** split of the **XL-Sum** dataset, comparing their performance via **ROUGE-1**, **ROUGE-2**, **ROUGE-L**, and **mBERT BERTScore**.

## Procedure & Implementation
This experiment compares two distinct multilingual seq2seq pretraining paradigms (span-corruption text-to-text vs. denoising auto-encoding) on Devanagari-script Hindi news summarization.

1.  **XL-Sum Dataset Preparation (`8.1`):**
    *   Downloading the official XL-Sum v2.0 Hindi archive (`hindi_XLSum_v2.0.tar.bz2`, BBC Hindi news articles) and extracting the test split <sup>(`hf_hub_download()`, `tarfile.open()`)</sup>.
    *   Analyzing article-to-summary compression ratios and sampling a fixed, reproducible evaluation subset (`N = 200`, `seed = 42`), persisted locally as `xlsum_hindi_test_sample.jsonl` <sup>(`DataFrame.sample()`, `to_json()`)</sup>.

2.  **Multilingual Evaluation Setup (`8.2`):**
    *   Implementing a **Unicode-aware `MultilingualTokenizer`** that strips Unicode punctuation and symbol categories (`P*`, `S*`, including Devanagari *purna viram* `।`) while preserving base letters (`Lo`) and combining vowel signs/matras (`Mn`, `Mc`) inside words <sup>(`unicodedata.category()`)</sup>.
    *   Demonstrating why standard `rouge_score` fails on Devanagari (its default regex `[^a-z0-9]+` strips all Hindi characters, yielding `0.0`) and configuring `RougeScorer` with the custom tokenizer <sup>(`rouge_scorer.RougeScorer()`)</sup>.

3.  **mT5 Abstractive Summarization (`8.3–8.4`):**
    *   Loading `csebuetnlp/mT5_multilingual_XLSum` (`~582M` parameters, `MT5ForConditionalGeneration` based on `google/mt5-base`, jointly fine-tuned across 45 XL-Sum languages including Hindi) with its `250,100`-token SentencePiece vocabulary <sup>(`AutoTokenizer.from_pretrained()`, `AutoModelForSeq2SeqLM.from_pretrained()`)</sup>.
    *   Generating summaries over the Hindi test subset using beam search (`num_beams=4`, `max_length=84`, `no_repeat_ngram_size=2`, input truncated to `512` tokens) <sup>(`model.generate()`, `batch_decode()`)</sup>.

4.  **mBART Abstractive Summarization (`8.5`):**
    *   Loading `ai4bharat/IndicBART-XLSum` (`~244M` parameters, `MBartForConditionalGeneration` with `6` encoder + `6` decoder layers and a `64,015`-token unified Devanagari SentencePiece vocabulary, fine-tuned on XL-Sum) <sup>(`_convert_token_to_id_with_added_voc('<2hi>')`)</sup>.
    *   Appending the Hindi language tag (`</s> <2hi>`) to each truncated input sequence and generating summaries under identical beam-search constraints with `decoder_start_token_id` forced to `<2hi>`.

5.  **Automatic Metric Computation & Statistical Comparison (`8.6`):**
    *   Computing lexical overlap metrics (**ROUGE-1**, **ROUGE-2**, **ROUGE-L** F1) using the Unicode-aware scorer and contextual semantic similarity (**BERTScore Precision, Recall, F1**) using `google-bert/bert-base-multilingual-cased` (mBERT) <sup>(`bert_score.score()`)</sup>.
    *   Conducting a paired per-article win-rate comparison with `2,000`-resample paired bootstrap `95%` confidence intervals and exporting all reference and model summaries to `8_generated_summaries.csv` <sup>(`np.percentile()`, `to_csv()`)</sup>.

## Observations
| Analysis Task | Key Observation Parameter | Visualization/Method |
| :--- | :--- | :--- |
| **Corpus Compression** | Article vs. summary length (`~13–14×` median compression) | Summary statistics & JSONL sample export |
| **Tokenizer Validation** | Default ASCII ROUGE (`0.0`) vs. Devanagari Unicode ROUGE | Side-by-side scorer comparison |
| **mT5 Generation** | Multilingual transfer across 45 languages (`582M` params) | Beam-search decoding (`beams=4`) |
| **mBART Generation** | Indic mBART denoising auto-encoder with `<2hi>` tag (`244M` params) | Beam-search decoding (`beams=4`) |
| **Lexical & Semantic Eval** | ROUGE-1/2/L & mBERT BERTScore distributions | Grouped bar chart, boxplots & bootstrap CI |

### Comparative Results (XL-Sum Hindi Test Sample, N = 200)
| Model | Checkpoint | Params | ROUGE-1 | ROUGE-2 | ROUGE-L | mBERT-P | mBERT-R | mBERT-F1 | Avg Gen Words (Ref: 27.5) | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **mT5** | `csebuetnlp/mT5_multilingual_XLSum` | 582.4M | **35.99** | **15.83** | **30.31** | **78.80** | **75.16** | **76.88** | 18.58 | 65.08s |
| **mBART** | `ai4bharat/IndicBART-XLSum` | **244.0M** | 20.52 | 6.56 | 17.61 | 74.07 | 67.26 | 70.44 | 9.40 | **22.67s** |
| **Paired Δ (`mBART − mT5`)** | *2,000-resample 95% bootstrap CI* | — | -15.47 `[-17.58, -13.29]` | -9.27 `[-11.17, -7.54]` | -12.70 `[-14.68, -10.65]` | -4.73 | -7.90 | -6.44 `[-7.21, -5.68]` | -9.18 | **2.87× faster** |

## Result Interpretation & Learnings
*   **Architecture & Parameter Efficiency (mT5 vs. mBART):** Comparing `mT5_multilingual_XLSum` (`582M` params, 45-language vocabulary of `250K` tokens) against `IndicBART-XLSum` (`244M` params, Indic-focused `MBartForConditionalGeneration` with a unified `64K` Devanagari vocabulary) demonstrates how script-unified mBART representations achieve strong Hindi summarization quality and faster inference at less than half the parameter count.
*   **Pretraining Objective Alignment:** Both mT5's span-corruption objective and mBART's full-sequence denoising auto-encoding (Poisson span masking + sentence permutation) effectively pretrain encoder-decoder cross-attention for abstractive compression, while mBART's explicit language token (`<2hi>`) prevents cross-lingual code-switching during decoding.
*   **Lexical Overlap (ROUGE) vs. Semantic Equivalence (mBERT):** Abstractive models frequently paraphrase Hindi headlines using synonymous postpositions, compound verbs, or reordered clauses that lower exact n-gram overlap (especially ROUGE-2) while preserving meaning—a distinction captured by mBERT contextual embeddings in **BERTScore**.
*   **Devanagari Evaluation Pitfalls:** Standard English-centric ROUGE pipelines silently return `0.0` on Hindi due to ASCII-only filtering (`[^a-z0-9]+`), and naive `\w+` regexes split words at spacing vowel matras (`Mc`); Unicode category filtering (`P*`/`S*`) is essential for accurate Indic NLP evaluation.
