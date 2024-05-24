import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np

# Descargar recursos de NLTK - Tokenizer y Corpus
nltk.download('punkt')
nltk.download('stopwords')

# Texto de ejemplo
texts = [
    "Natural Language Processing is fun and exciting!",
    "I love learning about machine learning.",
    "Text analysis can reveal interesting patterns."
]

# Función de preprocesamiento
def preprocess(text):
    text = text.lower()  # Convertir a minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    tokens = word_tokenize(text)  # Tokenizar
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Eliminar stop words
    return ' '.join(tokens)

# Preprocesar los textos
processed_texts = [preprocess(text) for text in texts]
print("Processed Texts:\n", processed_texts)

# Paso 4: Vectorización

# Paso 4.1: Bolsa de Palabras (Bag of Words)
# Crear el vectorizador de la bolsa de palabras
vectorizer = CountVectorizer()

# Transformar los textos
X = vectorizer.fit_transform(processed_texts)

# Mostrar el resultado de la bolsa de palabras
print("Bag of Words Representation:\n", X.toarray())
print("Feature Names:\n", vectorizer.get_feature_names_out(), "\n")

# Paso 4.2: TF-IDF
# Crear el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Transformar los textos
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)

# Mostrar el resultado de TF-IDF
print("TF-IDF (Bag of Words) Representation:\n", X_tfidf.toarray())
print("Feature Names:\n", tfidf_vectorizer.get_feature_names_out(), "\n")


# Paso 5: Cálculo de Frecuencias y Valores TF-IDF

# Sumar las frecuencias de las palabras en la representación de la bolsa de palabras
word_freq = np.sum(X.toarray(), axis=0)

# Obtener las palabras y sus frecuencias
words = vectorizer.get_feature_names_out()
word_freq_dict = dict(zip(words, word_freq))
print("Word Frequencies:\n", word_freq_dict, "\n")

# Obtener las palabras y sus valores TF-IDF
tfidf_values = np.sum(X_tfidf.toarray(), axis=0)
tfidf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_values))
print("TF-IDF Values:\n", tfidf_dict)