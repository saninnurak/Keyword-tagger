import re
import os
from collections import Counter
from itertools import chain
import nltk

# Download stopwords once
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
JUNK_WORDS = set(['000000', 'acct',])

def normalize(text):
    """Lowercase and remove punctuation"""
    return re.sub(r'[^\w\s]', '', text.lower())

def ngrams(tokens, n):
    """Generate n-grams from tokens"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def is_valid_phrase(phrase):
    """Filter: must contain a non-stopword and no junk words"""
    words = phrase.split()
    return (
        any(word not in STOPWORDS for word in words) and
        all(word not in JUNK_WORDS for word in words)
    )

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_keywords(sentences, min_freq=3, max_ngram=2):
    """Extract meaningful, frequent n-grams"""
    all_ngrams = []
    for sentence in sentences:
        tokens = normalize(sentence).split()
        for n in range(1, max_ngram + 1):
            ngram_list = ngrams(tokens, n)
            all_ngrams.extend([ng for ng in ngram_list if is_valid_phrase(ng)])
    phrase_counts = Counter(all_ngrams)
    return {phrase for phrase, count in phrase_counts.items() if count >= min_freq}

def tag_sentences(sentences, keywords):
    tagged = []
    for sentence in sentences:
        sentence_norm = normalize(sentence)
        tokens = sentence_norm.split()
        sentence_ngrams = set(chain.from_iterable(
            ngrams(tokens, n) for n in range(1, 4)
        ))

        # Match phrases from keywords
        matched = {kw for kw in keywords if kw in sentence_ngrams}

        # Prefer longer tags and remove substrings
        sorted_tags = sorted(matched, key=lambda x: -len(x.split()))
        final_tags = []
        for tag in sorted_tags:
            if not any(tag in longer for longer in final_tags):
                final_tags.append(tag)

        if final_tags:
            tagged.append(f"{sentence}\t{', '.join(final_tags)}")
        else:
            tagged.append(sentence)
    return tagged

def save_output(tagged_sentences, file_path='output/task_1_output.tsv'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in tagged_sentences:
            f.write(line + '\n')

if __name__ == '__main__':
    input_file = 'data/sentences.txt'
    if os.path.exists(input_file):
        sentences = load_sentences(input_file)
        keywords = extract_keywords(sentences, min_freq=4, max_ngram=3)
        tagged = tag_sentences(sentences, keywords)
        save_output(tagged, 'output/task_1_output.tsv')
        print(f"Tagged {len(sentences)} sentences with {len(keywords)} clean keyword tags.")
    else:
        print(f"Input file {input_file} not found.")
