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
