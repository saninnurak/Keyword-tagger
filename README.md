# Semantic and Rule-Based Keyword Tagger

This repository contains two approaches to sentence tagging based on a provided dataset :

1. **Task 1 - Rule-Based N-Gram Tagger**  
2. **Task 2 - Semantic Tagger using KeyBERT and SentenceTransformers**

Each method processes input sentences and produces a tab-separated output file with appropriate tags. These solutions meet the requirements of the assignment as outlined below.

---

## ⚙️ Installation

Follow the steps below to install and run the project:

1. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/saninnurak/Tag_extractor.git
    cd Keyword-tagger
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On **Windows**:

        ```bash
        venv\Scripts\activate
        ```

    - On **macOS/Linux**:

        ```bash
        source venv/bin/activate
        ```

4. **Install the required dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```
   
## 🚀 Running the Taggers

Once you've set up the project, you can run both the taggers as follows:

### Task 1 - Rule-Based N-Gram Tagger

To run the rule-based n-gram keyword tagger, use the following command:

```bash
python .\src\task_1_rule_based_key_extractor.py
```
This will generate a tab-separated file with tags for each sentence, based on the rule-based n-gram extraction.

### Task 2 - Semantic Tagger using KeyBERT and SentenceTransformers

To run the semantic tagger, use the following command:

```bash
python .\src\task_2_bert_keyword_extractor.py
```
This will generate a tab-separated file with semantic tags for each sentence, leveraging machine learning models for keyword extraction.


## 📁 Directory Structure
```plaintext
project/
├── data/
│   └── sentences.txt                       # Input sentences, one per line
│   └── tags.csv                            # Example of tags in csv
├── output/
│   ├── task_1_output.tsv                   # Output from rule-based tagger
│   └── task_2_output.tsv                   # Output from semantic tagger
├── task_1_rule_based_key_extractor.py      # Rule-based n-gram keyword tagger
├── task_2_bert_keyword_extractor.py        # Semantic tagging using ML
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```
## ⚖️ Key Differences

Below is a comparison of the two tagger scripts:

| **Aspect**               | **Script 1 (extract_keywords, tag_sentences)**            | **Script 2 (KeyBERT + SentenceTransformer)**                       |
|--------------------------|----------------------------------------------------------|--------------------------------------------------------------------|
| **Methodology**           | Rule-based, using n-grams, stopwords, frequency filtering| Embedding-based, semantic keyword extraction using KeyBERT         |
| **ML/AI Model**           | ❌ No ML — pure NLP rules                                | ✅ Yes — uses SentenceTransformer (MiniLM) with KeyBERT             |
| **Keyword Extraction Logic**| - Generate 1–3 n-grams <br> - Filter by frequency and junk words | - Uses transformer embeddings <br> - Ranks based on relevance      |
| **Stopwords**             | NLTK stopwords + manual junk words                       | KeyBERT stopwords + custom stopwords                               |
| **Tag Matching Logic**    | Matches extracted keywords directly from n-gram space    | Extracts most relevant keywords semantically using sentence meaning |
| **Code Style**            | More manual and deterministic                           | More abstract, less code but depends on heavy NLP libraries        |
| **Performance**           | Lightweight, fast                                       | Heavier due to model loading and embedding                         |
| **Output File**           | output/task_1_output.tsv                                | output/task_2_output.tsv                                           |
