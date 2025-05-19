import os, re
import numpy as np
import fitz, joblib
import sys 
from langdetect import detect, DetectorFactory
from tempfile import NamedTemporaryFile
from docx import Document
from deep_translator import GoogleTranslator
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

app = Flask(__name__)
app.secret_key = 'change_this_to_something_secret'

DetectorFactory.seed = 0
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, 'keras_lang_model.h5')
TFIDF_PATH        = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
SVD_PATH          = os.path.join(MODEL_DIR, 'svd_transformer.joblib')
ENCODER_PATH      = os.path.join(MODEL_DIR, 'label_encoder.joblib')

required_files = [
    'tfidf_vectorizer.joblib',
    'svd_transformer.joblib',
    'label_encoder.joblib',
    'keras_lang_model.h5'
]

missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_DIR, f))]

if missing:
    raise FileNotFoundError(
        f"Отсутствуют необходимые файлы в папке models/: {', '.join(missing)}.\n"
        f"Убедитесь, что папка 'models' находится рядом с main.exe и содержит все нужные модели." 
    )

keras_model = load_model(KERAS_MODEL_PATH)
tfidf       = joblib.load(TFIDF_PATH)
svd         = joblib.load(SVD_PATH)
y_encoder   = joblib.load(ENCODER_PATH)

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

def preprocess_text(text, label=None):
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r'[^\w\s]', ' ', text).lower()
    tokens = tokenizer.tokenize(cleaned)
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
            probs = keras_model.predict(X_red)[0]
            lang  = y_encoder.inverse_transform([int(probs.argmax())])[0]

            language_votes[lang] = language_votes.get(lang, 0) + 1
            processed_chunks += 1

    total = processed_chunks or 1
    return {lang: votes/total*100 for lang, votes in language_votes.items()}

def detect_language_from_text(text: str) -> str:
    if not text.strip():
        raise ValueError("Текст пустой или не извлечён.")
    # 1) предобработка
    proc = preprocess_text(text)  
    # 2) векторизация + редукция
    X_tfidf = tfidf.transform([proc])
    X_red   = svd.transform(X_tfidf)
    # 3) предсказание
    probs = keras_model.predict(X_red)[0]
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

@app.route('/', methods=['GET'])
def index():
    available = os.listdir(MODEL_DIR)
    return render_template(
        'index.html',
        supported_langs=SUPPORTED_LANGS,
        summary=None,            
        translations="",
        models = available          
    )

@app.route('/upload_model', methods=['POST'])
def upload_model():
    f = request.files.get('model_file')
    if not f:
        flash('Файл не выбран')
        return redirect(url_for('index'))
    
    filename = secure_filename(f.filename)
    save_path = os.path.join(MODEL_DIR, filename)
    f.save(save_path)

    try:
        global keras_model, tfidf, svd, y_encoder

        if filename.endswith('.h5'):
            keras_model = load_model(save_path)
            flash(f'Keras-модель {filename} успешно загружена')
        elif filename.endswith('.joblib'):
            if 'tfidf' in filename:
                tfidf = joblib.load(save_path)
                flash('TF-IDF векторизатор загружен')
            elif 'svd' in filename:
                svd = joblib.load(save_path)
                flash('SVD-трансформер загружен')
            elif 'encoder' in filename or 'label' in filename:
                y_encoder = joblib.load(save_path)
                flash('LabelEncoder загружен')
            else:
                flash(f'«{filename}» загружен, но не опознан как .joblib-модель')
        else:
            flash('Неподдерживаемый тип файла')
    except Exception as e:
        flash(f'Ошибка при загрузке модели: {e}')

    return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_file():
    # Инициализируем
    summary = None
    translations = ''
    tmp_name = None
    text = None
    
    # Если пользователь вставил текст вручную
    text_input = request.form.get('text_input', '').strip()
    if text_input:
        # При очень большом тексте - временный файл + process_large_file
        if len(text_input) > 500_000:
            tmp = NamedTemporaryFile(delete=False, suffix='.txt')
            tmp_name = tmp.name
            tmp.write(text_input.encode('utf-8'))
            tmp.close()
            summary = process_large_file(tmp_name, chunk_size=50_000)
        else:
            text = text_input

    else:
        # Иначе пытаемся загрузить файл
        f = request.files.get('file')
        if not f:
            return render_template(
                'index.html',
                supported_langs=SUPPORTED_LANGS,
                summary=summary,
                translations="Ошибка: ни текст, ни файл не переданы"
            ), 400

        suffix = os.path.splitext(secure_filename(f.filename))[1].lower()
        tmp = NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_name = tmp.name
        tmp.close()
        f.save(tmp_name)

        # Если это большой .txt - chunk-процессинг
        if suffix == '.txt' and os.path.getsize(tmp_name) > 5_000_000:
            summary = process_large_file(tmp_name, chunk_size=50_000)
        else:
            # Обычное извлечение текста
            try:
                text = extract_text_from_file(tmp_name)
            except Exception as e:
                os.unlink(tmp_name)
                return render_template(
                    'index.html',
                    supported_langs=SUPPORTED_LANGS,
                    summary=summary,
                    translations=f"Ошибка при чтении файла: {e}"
                ), 400
    # Для больших текстов мы уже могли получить summary через process_large_file
    # - в этом случае сразу рендерим
    if summary is not None:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except:
                pass
            return render_template(
                'index.html',
                supported_langs=SUPPORTED_LANGS,
                summary=summary,
                translations="translations"
            )
        
    # Для небольших текстов - детект по предложениям 
    if text:
        raw = re.split(r'(?<=[.!?])\s+|\r?\n+', text)
        sentences = [s.strip() for s in raw if s.strip()]
        counts = {}
        THRESHOLD = 100
        for sent in sentences:
            try:
                if len(sent) < THRESHOLD:
                    lang = detect(sent)
                else:
                    lang = detect_language_from_text(sent)
            except:
                lang = detect_language_from_text(sent)
            counts[lang] = counts.get(lang, 0) + 1
        total   = sum(counts.values()) or 1
        summary = {lang: cnt/total*100 for lang, cnt in counts.items()}

        # Перевод всего текста для выбранных языков
        targets = request.form.getlist('targets') or SUPPORTED_LANGS
        for tgt in targets:
            translations += f"--- {tgt} ---\n"
            for sent in sentences:
                try:
                    tr = GoogleTranslator(source='auto', target=tgt).translate(sent)
                except Exception as e:
                    tr = f"Ошибка перевода: {e}"
                translations += tr + "\n"
            translations += "\n"

        # Чистка
        if tmp_name:
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