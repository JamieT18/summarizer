# 📝 Powerful Text Summarizer

A modern web-based tool for generating **extractive** and **abstractive** summaries from text documents.  
Built with Flask and advanced NLP libraries, it offers an intuitive UI, cutting-edge summarization models, and smart accuracy enhancements.

---

## 🚀 Features

- **Extractive Summarization**
  - Ranks sentences by importance using semantic similarity (Sentence Transformers) or TF-IDF.
  - Displays sentence scores and lets you adjust summary length.

- **Abstractive Summarization**
  - Uses the latest transformer-based models (BART, Pegasus, T5).
  - Handles long texts via smart chunking and merging; applies a secondary summarization pass for coherence if needed.

- **Language Detection & Adaptation**
  - Auto-detects input language for adaptive stopword filtering in extractive summaries.

- **User-Friendly Web UI**
  - Paste text or upload `.txt` files.
  - Switch summarization method and model.
  - Tweak summary length, min/max output size.
  - See original and summarized text side by side.
  - View compression ratio and sentence importance.

- **Robust Input Handling**
  - Warns if input exceeds recommended limits, but still processes with best-effort chunking.
  - Error handling for file, model, or processing issues.

- **Performance & Security**
  - Caches models for faster response.
  - Limits input to 5000 words per submission to ensure stability and accuracy.

---

## 🛠 Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/JamieT18/summarizer.git
    cd summarizer
    ```

2. **Install dependencies**

    ```bash
    pip install flask transformers scikit-learn nltk torch sentence-transformers langdetect
    ```

3. **Download NLTK Data (if needed)**

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

---

## 🌐 Usage

1. **Run the Flask app**

    ```bash
    python app.py
    ```

2. **Open your browser**  
   Visit [http://localhost:5000](http://localhost:5000)

---

## ⚡ Recommended Input Limits

- **Abstractive:** Up to **500 words** per submission for best results. Longer texts are chunked and merged, but coherence may drop.
- **Extractive:** Up to **2000 words** for optimal ranking. Maximum allowed: **5000 words** (with warning).

---

## 📦 File Structure

```
summarizer/
├── app.py               # Flask web app
├── templates/
│   └── index.html       # Web UI template
└── README.md            # This file
```

---

## 🤖 Supported Models

- `facebook/bart-large-cnn` (default, fast and general-purpose)
- `google/pegasus-xsum` (state-of-the-art for news and technical writing)
- `t5-base` (flexible, multi-task)

Select your preferred model in the UI.

---

## 💡 Advanced Features

- **Semantic Extractive Summarization:**  
  Uses [sentence-transformers](https://www.sbert.net/) for deeper understanding and ranking.

- **Smart Abstractive Chunking:**  
  Sliding window with overlap for long texts; second-pass summarization for clarity.

- **Language Adaptation:**  
  Detects language with [langdetect](https://pypi.org/project/langdetect/) for better stopword filtering.

- **Compression Ratio & Sentence Scores:**  
  Get insights into summary efficiency and sentence importance.

- **Model Caching and Error Feedback:**  
  Faster response and clear UI messages for issues.

---

## 📝 Customization

- Add support for PDF/DOCX uploads.
- Integrate more summarization models.
- Enable multi-user authentication or deploy securely.

---

## 📄 License

MIT

---

## 👤 Author

[JamieT18](https://github.com/JamieT18)

---

## ❤️ Contributions

Pull requests and suggestions are welcome!
