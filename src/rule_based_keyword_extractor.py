"""
Optimized Rule-Based N-Gram Keyword Tagger

This script reads a list of sentences from a text file and extracts keywords
based on n-gram frequency. It includes the following steps:
- Normalize and tokenize each sentence
- Generate n-grams (1 to 3 words)
- Filter out stopwords, junk terms, and low-information n-grams
- Count n-gram frequencies and extract commonly used phrases
- Annotate each sentence with matching keyword tags (non-overlapping)

This version includes smarter filters for low-quality tokens, improved de-duplication
of tags, and better handling of short words and digits.

Use when: You want fast, interpretable keyword tagging without ML dependencies.

Output: `output/task_1_output321.tsv`
"""

import re
import os
from collections import Counter
from itertools import chain
import nltk

# Download stopwords once
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
JUNK_WORDS = set(['000000', 'acct'])
MIN_TOKEN_LENGTH = 3

def normalize(text):
    """Lowercase and remove punctuation"""
    return re.sub(r'[^\w\s]', '', text.lower())

def tokenize(text):
    return [tok for tok in text.split() if len(tok) >= MIN_TOKEN_LENGTH and not tok.isdigit()]

def ngrams(tokens, n):
    """Generate n-grams from tokens"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def is_valid_phrase(phrase):
    """Filter: must contain at least one non-stopword and no junk words"""
    words = phrase.split()
    return (
        any(w not in STOPWORDS for w in words) and
        all(w not in JUNK_WORDS for w in words)
    )

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_keywords(sentences, min_freq=3, max_ngram=3):
    """Extract meaningful, frequent n-grams"""
    all_ngrams = []
    for sentence in sentences:
        tokens = tokenize(normalize(sentence))
        for n in range(1, max_ngram + 1):
            ngram_list = ngrams(tokens, n)
            all_ngrams.extend([ng for ng in ngram_list if is_valid_phrase(ng)])
    phrase_counts = Counter(all_ngrams)
    return {phrase for phrase, count in phrase_counts.items() if count >= min_freq}

def tag_sentences(sentences, keywords):
    tagged = []
    for sentence in sentences:
        tokens = tokenize(normalize(sentence))
        sentence_ngrams = set(chain.from_iterable(
            ngrams(tokens, n) for n in range(1, 4)
        ))

        # Match phrases from keywords
        matched = {kw for kw in keywords if kw in sentence_ngrams}

        # Prefer longer tags and remove substrings
        sorted_tags = sorted(matched, key=lambda x: (-len(x.split()), -len(x)))
        final_tags = []
        for tag in sorted_tags:
            if not any(tag in longer and tag != longer for longer in final_tags):
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
