"""
Rule-Based Tagging Script (Task 1)
----------------------------------

This script implements a simple rule-based tagging system for sentences using predefined keyword-tag mappings.

Functionality:
- Loads sentence data from 'data/sentences.txt'.
- Loads tags and associated keywords from 'data/tags.csv'.
- For each sentence, it checks for presence of any tag-related keywords (case-insensitive match).
- Outputs results in 'output/task_1_output.tsv' with format:
    sentence<TAB>tag1, tag2, tag3

File structure:
- Input:
    - data/sentences.txt — one sentence per line
    - data/tags.csv — CSV file with columns: id, name, keywords (Python list)
- Output:
    - output/task_1_output.tsv — tab-separated file with 'sentence' and 'tags' columns

Example output line:
    The system integrates AI and automation.    Artificial Intelligence, Automation

This rule-based approach is fast but limited to exact or substring keyword matches.

Usage:
    python task_1_tag_sentence_mapper.py

"""


import csv
import ast
import time

from datetime import datetime

# Helper to print with timestamp
def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Load tags and keywords
def load_tags(path):
    log(f"Loading tags from '{path}'...")
    tags = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                keyword_list = ast.literal_eval(row['keywords'])
                tags.append({
                    'id': row['id'],
                    'name': row['name'],
                    'keywords': set(kw.lower() for kw in keyword_list)  # set for faster lookup
                })
            except Exception as e:
                log(f"Error parsing row {row['id']}: {e}")
    log(f"Loaded {len(tags)} tags.")
    return tags

# Load sentences from file
def load_sentences(path):
    log(f"Loading sentences from '{path}'...")
    with open(path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    log(f"Loaded {len(sentences)} sentences.")
    return sentences

# Match tags to sentences
def tag_sentences(sentences, tags):
    log("Tagging sentences...")
    start_time = time.time()
    results = []
    for i, sentence in enumerate(sentences, start=1):
        sentence_lower = sentence.lower()
        matched = [
            tag['name'] for tag in tags
            if any(keyword in sentence_lower for keyword in tag['keywords'])
        ]
        results.append({'sentence': sentence, 'tags': ', '.join(matched)})
        if i % 100 == 0:
            log(f"Processed {i} sentences...")
    log(f"Tagging completed. Total: {len(results)}. Time: {time.time() - start_time:.2f} seconds.")
    return results

# Save to TSV
def save_results(results, output_path):
    log(f"Saving results to '{output_path}'...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sentence', 'tags'], delimiter='\t')
        writer.writerows(results)
    log("Results saved successfully.")

# Main pipeline
def main():
    start = time.time()
    tags = load_tags('data/tags.csv')
    sentences = load_sentences('data/sentences.txt')
    results = tag_sentences(sentences, tags)
    save_results(results, 'output/task_1_output.tsv')
    log(f"Finished entire process in {time.time() - start:.2f} seconds.")

if __name__ == '__main__':
    main()
