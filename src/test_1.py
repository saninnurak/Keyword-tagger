from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import os
import string

# Optional: domain-specific stopwords
custom_stopwords = set([
    'account', 'bank', 'aiqfinancial', 'checking', 'savings', 'hello',
    'thanks', 'thank', 'please', 'hi', 'help', 'need', 'information'
])

def clean_text(text):
    # Lowercase, remove punctuation
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Load sentences
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# Extract tags using KeyBERT with enhancements
def extract_tags(sentences, top_n=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model)
    results = []
    for sentence in sentences:
        keywords = kw_model.extract_keywords(
            sentence,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=top_n
        )
        # Filter tags
        tags = [kw for kw, _ in keywords if kw not in custom_stopwords]
        results.append((sentence, tags))
    return results

# Save output to TSV
def save_to_tsv(tagged_sentences, file_path='task_1_output.tsv'):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence, tags in tagged_sentences:
            if tags:
                line = f"{sentence}\t{', '.join(tags)}"
            else:
                line = sentence
            f.write(line + '\n')

if __name__ == '__main__':
    input_file = 'data/sentences.txt'
    if os.path.exists(input_file):
        sentences = load_sentences(input_file)
        tagged = extract_tags(sentences)
        save_to_tsv(tagged)
        print("Improved tags extracted and saved to task_1_output.tsv")
    else:
        print(f"Input file {input_file} not found.")
