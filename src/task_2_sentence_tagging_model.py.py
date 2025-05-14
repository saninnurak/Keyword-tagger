from typing import List, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import numpy as np
import os

def load_data(tags_path: str, sentences_path: str) -> Tuple[pd.DataFrame, List[str]]:
    task_df = pd.read_csv(tags_path)
    task_df['keywords'] = task_df['keywords'].apply(eval)
    with open(sentences_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return task_df, sentences

def build_training_data(task_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [(kw.lower(), row['name']) for _, row in task_df.iterrows() for kw in row['keywords']]
    train_df = pd.DataFrame(pairs, columns=['sentence', 'tag'])
    return train_df.groupby('sentence')['tag'].apply(list).reset_index()

def encode_labels(tags: pd.Series) -> Tuple[MultiLabelBinarizer, np.ndarray]:
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(tags)
    return mlb, y

def generate_embeddings(model_name: str, sentences: List[str]) -> np.ndarray:
    print(f"🔄 Generating embeddings with: {model_name}...")
    model = SentenceTransformer(model_name)
    return model.encode(sentences, show_progress_bar=True)

def build_nn_model(input_dim: int, output_dim: int) -> Sequential:
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')  # sigmoid for multilabel
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model_nn(X: np.ndarray, y: np.ndarray) -> Sequential:
    model = build_nn_model(X.shape[1], y.shape[1])
    print("🧠 Training neural network...")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=1)
    return model

def predict_tags_nn(model, X_test: np.ndarray, mlb: MultiLabelBinarizer, threshold: float) -> List[List[str]]:
    print(f"🔍 Predicting with threshold {threshold}...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= threshold).astype(int)
    return mlb.inverse_transform(y_pred)

def save_predictions(sentences: List[str], predictions: List[List[str]], output_path: str):
    output_lines = []
    predicted_count = 0
    for sentence, tags in zip(sentences, predictions):
        line = f"{sentence}\t{', '.join(tags)}" if tags else f"{sentence}\t"
        output_lines.append(line)
        if tags:
            predicted_count += 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print("\n🔍 Preview of first 10 predictions:")
    for line in output_lines[:10]:
        print(line)
    print(f"\n✅ Total sentences with predictions: {predicted_count}")
    print(f"💾 Predictions saved to: {output_path}")

def main(threshold: float = 0.2):
    tags_path = 'data/tags.csv'
    sentences_path = 'data/sentences.txt'
    output_path = 'output/task_2_output.tsv'
    model_name = 'paraphrase-MiniLM-L6-v2'

    task_df, sentences = load_data(tags_path, sentences_path)
    grouped_train = build_training_data(task_df)

    mlb, y_train = encode_labels(grouped_train['tag'])
    X_train = generate_embeddings(model_name, grouped_train['sentence'].tolist())
    X_test = generate_embeddings(model_name, sentences)

    model = train_model_nn(X_train, y_train)
    predicted_tags = predict_tags_nn(model, X_test, mlb, threshold)

    save_predictions(sentences, predicted_tags, output_path)

if __name__ == "__main__":
    main(threshold=0.3)  # Lower threshold for broader predictions