import os, re
import numpy as np
import fitz, joblib
from tempfile import NamedTemporaryFile
from docx import Document
from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator
from tensorflow.keras.models import load_model
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

LABEL_TO_ISO = {
    'ru': 'ru',
    'de': 'de',
    'tr': 'tr',
    'el': 'el',
    'en': 'en',
    'pt': 'pt',
    'fr': 'fr',
    'it': 'it',
    'es': 'es'
}

tfidf     = joblib.load('tfidf_vectorizer.joblib')
svd       = joblib.load('svd_transformer.joblib')
y_encoder = joblib.load('label_encoder.joblib')
model     = load_model('keras_lang_model.h5')

tokenizer = WordPunctTokenizer()

LANG_MAP_NLTK = {
    'ru':'russian', 'de':'german',
    'tr':'turkish', 'el':'greek',
    'en':'english', 'pt':'portuguese',
    'fr':'french', 'it':'italian',
    'es':'spanish'
}

STOPWORDS_MAP = {lbl:set(stopwords.words(lang)) for lbl,lang in LANG_MAP_NLTK.items()}
STEMMER_MAP   = {lbl:SnowballStemmer(lang) for lbl,lang in LANG_MAP_NLTK.items() if lang in SnowballStemmer.languages}

def preprocess_text(text, label):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    tokens = tokenizer.tokenize(text)
    sw = STOPWORDS_MAP.get(label, STOPWORDS_MAP.get('en', set()))
    tokens = [t for t in tokens if t not in sw]
    stemmer = STEMMER_MAP.get(label)
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

SUPPORTED_LANGS = list(y_encoder.classes_)

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif ext == '.docx':
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Формат файла {ext} не поддерживается.")

def process_large_file(file_path, chunk_size=10000):
    language_votes = {}
    processed_chunks = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            proc = preprocess_text(chunk, label='en')
            X_vec = tfidf.transform([proc])
            X_red = svd.transform(X_vec)
            probs = model.predict(X_red)[0]
            lang  = y_encoder.inverse_transform([int(probs.argmax())])[0]

            language_votes[lang] = language_votes.get(lang, 0) + 1
            processed_chunks += 1

    total = processed_chunks or 1
    return {lang: votes/total*100 for lang, votes in language_votes.items()}

def detect_language_from_text(text: str) -> str:
    if not text.strip():
        raise ValueError("Текст пустой или не извлечён.")
    # 1) предобработка
    proc = preprocess_text(text, label='en')  
    # 2) векторизация + редукция
    X_tfidf = tfidf.transform([proc])
    X_red   = svd.transform(X_tfidf)
    # 3) предсказание
    probs = model.predict(X_red)[0]
    idx   = int(np.argmax(probs))
    return y_encoder.inverse_transform([idx])[0]

def detect_language_from_file(file_path: str) -> str:
    text = extract_text_from_file(file_path)
    return detect_language_from_text(text)

def translate_text(text: str, source_label: str, target_labels: list) -> dict:
    translations = {}

    src_iso = LABEL_TO_ISO.get(source_label)
    if src_iso is None:
        raise ValueError(f"Unknown source language label: {source_label}")

    for tgt_label in target_labels:
        tgt_iso = LABEL_TO_ISO.get(tgt_label)
        if tgt_iso is None:
            translations[tgt_label] = "Unsupported target label"
            continue
        try:
            translations[tgt_label] = GoogleTranslator(
                source=src_iso,
                target=tgt_iso
            ).translate(text)
        except Exception as e:
            translations[tgt_label] = f"Ошибка перевода: {e}"

    return translations

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        supported_langs=SUPPORTED_LANGS,
        summary=None,            
        translations=""          
    )

@app.route('/process', methods=['POST'])
def process_file():
    f = request.files.get('file')
    if not f:
        return render_template(
            'index.html',
            supported_langs=SUPPORTED_LANGS,
            summary=None,
            translations="Ошибка: файл не передан"
        ), 400

    suffix = os.path.splitext(secure_filename(f.filename))[1]
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_name = tmp.name
    tmp.close()
    f.save(tmp_name)
    
    ext = suffix.lower()
    if ext == '.txt' and os.path.getsize(tmp_name) > 5_000_000:
        summary = process_large_file(tmp_name, chunk_size=50_000)
        translations = ''
    else:
        text = extract_text_from_file(tmp_name)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        counts = {}
        for sent in sentences:
            lang = detect_language_from_text(sent)
            counts[lang] = counts.get(lang, 0) + 1
        total   = len(sentences) or 1
        summary = {lang: cnt/total*100 for lang, cnt in counts.items()}
        targets = request.form.getlist('targets') or SUPPORTED_LANGS
        translations = ""
        for tgt in targets:
            try:
                tr = GoogleTranslator(source='auto', target=tgt).translate(text)
            except Exception as e:
                tr = f"Ошибка перевода: {e}"
            translations += f"--- {tgt} ---\n{tr}\n\n"
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

    return render_template(
        'index.html',
        supported_langs=SUPPORTED_LANGS,
        summary=summary,
        translations=translations
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)