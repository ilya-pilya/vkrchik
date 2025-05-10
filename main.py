import os
import webbrowser
import sys
from threading import Timer
from joblib import load
from tensorflow.keras.models import load_model
from app import app

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'models')

model_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"[main.py] Не найден: {model_path}")

def open_browser():
    try:
        webbrowser.open_new('http://localhost:5000/')
    except Exception as e:
        print(f'Не удалось автоматически открыть браузер: {e}')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False, host='localhost', port=5000, use_reloader=False)
