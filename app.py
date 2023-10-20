import re

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

document_terms = {}


def updateSet(document_terms):
    terms_set = set()
    # Recorre los valores (arreglos de términos) en el diccionario y agrega los términos al conjunto
    for terminos in document_terms.values():
        terms_set.update(terminos)
    return terms_set


# Función para rastrear y extraer títulos
def crawl_and_extract_titles(url, max_levels, current_level, document_terms):
    if current_level > max_levels:
        return

    # Realizar la solicitud GET
    full_url = "https://es.wikipedia.org" + url
    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"Error al acceder a {full_url}")
        return

    # Parsear la página
    soup = BeautifulSoup(response.text, "html.parser")

    # Extraer títulos H1
    titles = [h1.text for h1 in soup.find_all("p")]
    document_terms[url] = titles

    # Extraer enlaces dentro de párrafos
    links = []
    for paragraph in soup.find_all("p"):
        links.extend(
            [
                link.get("href")
                for link in paragraph.find_all("a")
                if link.get("href") is not None
            ]
        )

    # Filtrar enlaces que cumplan con las condiciones
    valid_links = [
        link
        for link in links
        if link.startswith("/wiki/") and link not in document_terms
    ]

    # Limitar la cantidad de enlaces por nivel a 10
    valid_links = valid_links[:10]

    # Recursivamente rastrear los enlaces
    for link in valid_links:
        crawl_and_extract_titles(link, max_levels, current_level + 1, document_terms)


# Función para obtener los 10 documentos más similares dentro de un grupo
def documentos_similares_en_grupo(
    group_label, tfidf_matrix, consulta_vector, n_similares=10
):
    # Filtra los documentos que pertenecen al grupo especificado
    documentos_grupo = tfidf_matrix[tfidf_df["Grupo"] == group_label]

    # Calcula la similitud coseno entre la consulta y todos los documentos en el grupo
    similitudes = cosine_similarity(consulta_vector, documentos_grupo)

    # Ordena los documentos por similitud en orden descendente
    documentos_ordenados = similitudes[0].argsort()[::-1]

    # Obtiene los índices de los 10 documentos más similares
    top_n_similares = documentos_ordenados[:n_similares]

    return top_n_similares


def crawl():
    initial_url = "/wiki/Ciencias_de_la_computaci%C3%B3n"

    # Llamar a la función de rastreo
    crawl_and_extract_titles(
        initial_url, max_levels=2, current_level=0, document_terms=document_terms
    )

    nltk.download("punkt")

    # Recorrer los términos de cada documento en document_terms
    for document_url, terms in document_terms.items():
        tokenized_terms = []  # Lista para almacenar los términos tokenizados
        for term in terms:
            term = re.sub(r"[^\w\s]", "", term)
            term = re.sub(r"\d", "", term)
            if term:
                tokens = word_tokenize(
                    term.lower()
                )  # Convertir a minúsculas para la normalización
                tokenized_terms.extend(tokens)  # Agregar tokens a la lista

        document_terms[
            document_url
        ] = tokenized_terms  # Reemplazar la lista de términos por los tokens

    nltk.download("stopwords")

    # Obtener una lista de palabras vacías en español
    stop_words = set(stopwords.words("spanish"))

    # Recorrer y reemplazar los términos de cada documento en document_terms
    for document_url, terms in document_terms.items():
        # Filtrar términos que no son palabras vacías
        filtered_terms = [term for term in terms if term not in stop_words]
        document_terms[
            document_url
        ] = filtered_terms  # Reemplazar la lista de términos por los términos filtrados

    # Crear un objeto SnowballStemmer para español
    stemmer = SnowballStemmer("spanish")

    # Recorrer y reemplazar los términos de cada documento en document_terms con stemming
    for document_url, terms in document_terms.items():
        # Aplicar stemming a los términos
        stemmed_terms = [stemmer.stem(term) for term in terms]
        document_terms[
            document_url
        ] = stemmed_terms  # Reemplazar la lista de términos por los términos con stemming


crawl()


@app.route("/query", methods=["POST"])
def query():
    consulta_q = request.json.get("query")
    k = request.json.get("k")

    nltk.download("stopwords")
    stop_words = set(stopwords.words("spanish"))

    stemmer = SnowballStemmer("spanish")

    # Aplicar eliminación de palabras vacías (stopwords) y stemming a la consulta Q
    def preprocess_query(query):
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        return " ".join(stemmed_words)

    # Aplicar la preprocesamiento a la consulta Q
    consulta_procesada = preprocess_query(consulta_q)

    # Crea una lista con la consulta Q
    consulta_q = [consulta_procesada]

    # Calcula el vector de la consulta Q utilizando el mismo vectorizador TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documentos)  # Ajusta el vectorizador al corpus existente
    vector_q = vectorizer.transform(consulta_q)

    # Convierte el vector de la consulta Q a un DataFrame de Pandas
    vector_q_df = pd.DataFrame(
        vector_q.toarray(), columns=vectorizer.get_feature_names_out()
    )

    #################################3
    documentos = [" ".join(terminos) for terminos in document_terms.values()]

    # Calcula la matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documentos)

    # Convierte la matriz TF-IDF a un DataFrame de Pandas
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )
    tfidf_df.insert(0, "URL", list(document_terms.keys()))

    ###################################################

    # Agrega la consulta Q al conjunto de documentos
    documentos.append(consulta_procesada)

    # Actualiza la matriz TF-IDF para incluir la consulta Q
    tfidf_matrix = tfidf_vectorizer.transform(documentos)

    # Convierte la matriz TF-IDF actualizada a un DataFrame de Pandas
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()
    )
    document_keys = list(document_terms.keys())
    document_keys.append("consulta")
    tfidf_df.insert(0, "URL", document_keys)

    #############################################

    # Crea un modelo K-means con k clusters
    kmeans = KMeans(n_clusters=k, random_state=0)

    # Ajusta el modelo K-means a los datos (matriz TF-IDF)
    kmeans.fit(tfidf_matrix)

    # Obtiene las etiquetas de grupo asignadas a cada documento
    etiquetas_grupos = kmeans.labels_

    # Agrega las etiquetas de grupos al DataFrame de documentos
    tfidf_df["Grupo"] = etiquetas_grupos

    ########################################################

    # Número de documentos más similares a mostrar
    n_similares = 10

    # Calcula los 10 documentos más similares a la consulta Q en su grupo
    grupo_consulta_q = tfidf_df.loc[tfidf_df["Grupo"] == etiquetas_grupos[-1]]
    documentos_similares_indices = documentos_similares_en_grupo(
        etiquetas_grupos[-1], tfidf_matrix, vector_q, n_similares
    )

    resultados = grupo_consulta_q.iloc[documentos_similares_indices[1:]][
        "URL"
    ].values.tolist()

    return jsonify(resultados)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
