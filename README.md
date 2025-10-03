## Overview
This Python project extracts text from PDFs and generates concise summaries using **pre-trained word embeddings** and sentence similarity. It helps students quickly understand long educational PDFs.

---

## Features
- Extracts text from PDFs using **PyPDF2**.
- Filters stopwords and punctuation.
- Converts sentences to vectors using **GloVe embeddings**.
- Ranks sentences based on similarity and outputs the top sentences as a summary.

---

## Requirements
- Python 3.13+
- Libraries: `PyPDF2`, `nltk`, `numpy`
- Pre-trained **GloVe word embeddings** (keep locally; do not push to GitHub)
- # Setup
1. Download GloVe embeddings: https://nlp.stanford.edu/projects/glove/
2. Place the file `glove.6B.50d.txt` in the project folder.
3. Run `python app.py`


---

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EduVision-Windsor-Chatgpt.git
cd EduVision-Windsor-Chatgpt
