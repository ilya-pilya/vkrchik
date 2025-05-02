import joblib

v = joblib.load('tfidf_vectorizer.pkl')
print('idf_ found?', hasattr(v, 'idf_'))
