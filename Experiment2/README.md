# Experiment 2: Basic Text Preprocessing

**Name:** Aantriksh Sood  
**SAP ID:** 500124259  

## Objective
To implement foundational text preprocessing techniques essential for Natural Language Processing by transforming raw textual data into structured tokens while removing noise.

## Procedure & Implementation
The experiment operates on raw text files and is divided into three core preprocessing stages, leveraging built-in Python string operations, NLTK, and spaCy libraries.

1.  **Text Cleaning (`2.1_text_data.txt`):**
    *   Reading the raw text file.
    *   Calculating the frequency of uppercase characters.
    *   Converting the entire text corpus to lowercase for uniformity.
    *   Identifying and stripping all punctuation marks using the `string.punctuation` module.
    *   Removing numeric digits.
    *   Normalizing whitespace by splitting and rejoining the text, calculating the total whitespace characters removed.

2.  **Tokenization (`2.2_tokenization_data.txt`):**
    *   **NLTK:** Utilizing `nltk.word_tokenize` and `nltk.sent_tokenize` for word and sentence level tokenization.
    *   **spaCy:** Loading the `en_core_web_sm` model to process the text and extract word and sentence tokens via the document object.
    *   **Python Native:** Implementing basic tokenization using the `split()` method for words and regular expressions (`re.split`) for sentence boundary detection.

3.  **Stop Words Removal (`2.3_clean_data.txt`):**
    *   Downloading and importing the English stop words list from `nltk.corpus`.
    *   Tokenizing the lowercase input text.
    *   Filtering the token list to exclude any words present in the stop words set and ensuring only alphanumeric tokens remain.
    *   Isolating and displaying the specific stop words that were removed from the corpus.

## Result Interpretation & Learnings
*   **Pipeline Foundation:** The experiment practically demonstrates the critical initial phases necessary for natural language processing pipelines.
*   **Effective Normalization:** The text cleaning stage successfully normalizes raw text by eliminating non-informative characters, including punctuation, digits, and excess whitespace.
*   **Tokenizer Comparison:** The tokenization phase highlights syntactical differences and operational efficiencies between rule-based approaches (NLTK, regex) and model-based architectures (spaCy).
*   **Dimensionality Reduction:** The stop word removal process effectively reduces the dimensionality of the text data corpus.
*   **Semantic Isolation:** Post preprocessing, only the semantically significant tokens remain, perfectly formatted for downstream computational linguistics tasks.
