import streamlit as st
import joblib
import os

st.set_page_config(page_title='Определение языка текста', layout='centered')

model_path = 'language_classifier.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

@st.cache_resource
def load_model_and_vectorizer():
    if not os.path.exists(model_path):
        st.error(f'Файл модели не найден: {model_path}')
        return None, None
    if not os.path.exists(vectorizer_path):
        st.error(f'Файл векторизатора не найден: {vectorizer_path}')
        return None, None
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.title('Определение языковой группы текста')
st.write('Введите слово или фразу, и модель определит язык: немецкий **DE**, греческий **GR**, турецкий **TR** или русский **RU**')

user_input = st.text_input('Введите текст:')

if user_input and model and vectorizer:
    X_input = vectorizer.transform([user_input])
    prediction = model.predict(X_input)[0]

    lang_map = {
        'ger': 'Немецкий 🇩🇪 (DE)',
        'gre': 'Греческий 🇬🇷 (GR)',
        'tur': 'Турецкий 🇹🇷 (TR)',
        'rus': 'Русский 🇷🇺 (RU)',
    }

    st.success(f'**Предсказанный язык:** {lang_map.get(prediction, prediction)}')