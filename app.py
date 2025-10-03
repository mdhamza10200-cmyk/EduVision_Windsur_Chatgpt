import string
import numpy as np
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance

stop_words = set(stopwords.words('english'))

# Read PDF text
def read_pdf_file(file_path):
    text = ''
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PdfReader(pdf_file_obj)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
    return text

# Load GloVe embeddings
word_embeddings = {}
with open('C:\EduVision Windsor 1\word_embeddings.txt', 'r', encoding='utf8') as f:  # replace with your GloVe file
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.asarray(values[1:], dtype=float)
        word_embeddings[word] = vector

dim = len(next(iter(word_embeddings.values())))  # dynamic vector size

# Extract summary
def extract_summary(text):
    # Clean text
    text = text.replace('\n', ' ').replace('"', '')
    
    sentences = text.split('.')
    clean_sentences = [s.lower().strip() for s in sentences if s.strip()]

    sentence_vectors = []
    valid_sentences = []

    for sentence in clean_sentences:
        words = [w.strip(string.punctuation) for w in sentence.split()]
        filtered_words = [w for w in words if w not in stop_words and w in word_embeddings]

        if not filtered_words:
            continue

        vector = np.sum([word_embeddings[w] for w in filtered_words], axis=0)
        sentence_vectors.append(vector)
        valid_sentences.append(sentence)

    if not sentence_vectors:
        return ""

    n = len(sentence_vectors)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = 1 - cosine_distance(sentence_vectors[i], sentence_vectors[j])

    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = sorted(
        ((score, sent) for score, sent in zip(sentence_scores, valid_sentences)),
        reverse=True
    )

    summary = " ".join([s for _, s in ranked_sentences[:5]])
    return summary

if __name__ == '__main__':
    file_path =  "C:/EduVision Windsor 1/Abstract.pdf"  # replace with your PDF
    text = read_pdf_file(file_path)
    summary = extract_summary(text)
    print(summary)
