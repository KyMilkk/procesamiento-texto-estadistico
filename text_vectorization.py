import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
import matplotlib.pyplot as plt

# Descargar recursos de NLTK
nltk.download('punkt')       # Descargar el tokenizer Punkt
nltk.download('stopwords')   # Descargar las stopwords (palabras vacías)

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
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Eliminar stopwords
    return ' '.join(tokens)

# Preprocesar los textos
processed_texts = [preprocess(text) for text in texts]

# Crear el vectorizador de la bolsa de palabras
vectorizer = CountVectorizer()
# Transformar los textos
X = vectorizer.fit_transform(processed_texts)

# Crear el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()
# Transformar los textos
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts)

# Obtener las palabras y sus frecuencias
words = vectorizer.get_feature_names_out()
word_freq = np.sum(X.toarray(), axis=0)

# Obtener las palabras y sus valores TF-IDF
tfidf_values = np.sum(X_tfidf.toarray(), axis=0)


# Visualización con Matplotlib
plt.figure(figsize=(12, 6))

# Gráfico de barras para frecuencias de palabras
plt.subplot(1, 2, 1)
plt.bar(words, word_freq)
plt.title('Word Frequencies')
plt.xlabel('Words')
plt.ylabel('Frequency')

# Gráfico de barras para valores TF-IDF
plt.subplot(1, 2, 2)
plt.bar(words, tfidf_values)
plt.title('TF-IDF Values')
plt.xlabel('Words')
plt.ylabel('TF-IDF Value')

plt.tight_layout()  # Ajustar el diseño de los gráficos
plt.show()