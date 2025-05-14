# 🏷️ Sentence Tagging System (Rule-Based & Semantic ML)

This project performs **multi-label classification** on sentences using two complementary approaches:

- ✅ **Task 1:** Rule-based tag assignment using keyword and n-gram matching  
- ✅ **Task 2:** Semantic tagging with sentence embeddings and a neural network

---

## ⚙️ Installation

Follow the steps below to install and run the project:

1. **Ensure Python Version 3.11.9 is Installed**  
   This project requires **Python 3.11.9**. To check your Python version, use the following command:

    ```bash
    python --version
    ```

   If you're using a different version of Python, please install **Python 3.11.9** for compatibility.

2. **Clone this repository** to your local machine:

    ```bash
    git clone https://github.com/saninnurak/Keyword-tagger.git
    cd Keyword-tagger
    ```

3. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    ```

4. **Activate the virtual environment:**

    - On **Windows**:

        ```bash
        venv\Scripts\activate
        ```

    - On **macOS/Linux**:

        ```bash
        source venv/bin/activate
        ```

5. **Install the required dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## 🚀  How to Run

Once you've set up the project, you can run both the taggers as follows:

### ✅ Task 1


```bash
python .\src\task_1_tag_sentence_mapper.py
```

This method matches n-grams in input sentences to predefined keywords listed in data/tags.csv

### Task 2

To run the semantic tagger, use the following command:

```bash
python .\src\task_2_sentence_tagging_model.py
```

Optionally, you can adjust the confidence threshold for tag prediction (default is 0.3):
```bash
python .\src\task_2_sentence_tagging_model.py --threshold 0.3
```

This method embeds sentences and tags using sentence-transformers and trains a neural network for semantic multi-label classification.

## 📁 Directory Structure

```plaintext
project/
├── data/
│   └── sentences.txt                       # Input sentences, one per line
│   └── tags.csv                            # Example of tags in csv
├── output/
│   ├── task_1_output.tsv                   # Output from rule-based tagger
│   └── task_2_output.tsv                   # Output from semantic tagger
├── task_1_tag_sentence_mapper.py           # Rule-based n-gram keyword tagger
├── task_2_sentence_tagging_model.py        # Semantic tagging using ML
├── Notebook for prototyping.ipynb          # Jupyter notebook with prototypes 
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

## ## ⚖️ Key Differences

Below is a comparison of the two tagging approaches implemented in this project:

| **Aspect**                   | **Task 1: Rule-Based Tagger**                                  | **Task 2: Semantic ML Tagger**                                      |
|------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------|
| **Script**                   | `task_1_tag_sentence_mapper.py`                                 | `task_2_sentence_tagging_model.py`                                   |
| **Methodology**              | Keyword matching using n-grams and predefined tag keywords       | Sentence embeddings + neural network for tag classification          |
| **ML/AI Model**              | ❌ No — purely deterministic using rules                         | ✅ Yes — uses SentenceTransformer + Keras neural network              |
| **Input Data**               | `tags.csv` (keywords per tag) + `sentences.txt`                 | Same                                                               |
| **Text Representation**      | Plain text, n-gram generation                                   | Semantic vector representation using `paraphrase-MiniLM-L6-v2`       |
| **Keyword/Tag Matching**     | Matches lowercase n-grams against keyword lists                 | Predicts tags based on sentence meaning                             |
| **Label Handling**           | One or multiple tags per sentence (if keywords match)           | Multi-label classification with threshold                           |
| **Stopwords Handling**       | NLTK stopwords + custom junk filter                             | Implicit — handled by sentence embedding model                      |
| **Performance**              | ⚡ Fast, minimal dependencies                                    | 🧠 Slower (model loading & training) but more flexible               |
| **Output File**              | `output/task_1_output.tsv`                                      | `output/task_2_output.tsv`                                          |
| **Best Used For**            | Simple, deterministic use cases with clear keyword-tag mapping  | Complex language understanding and flexible tag inference            |

## 🙋‍♂️ Maintainer

**Sanin Nurak**  
📧 [saninnurak@hotmail.com](mailto:saninnurak@hotmail.com)  
🔗 [github.com/saninnurak](https://github.com/saninnurak)
