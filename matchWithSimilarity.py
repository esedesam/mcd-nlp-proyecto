import pandas as pd
import spacy
import re
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def normalizeDoc(nlp, doc, nMinCharacters = 4):
    """
    Normaliza un texto eliminando palabras por debajo del mínimo de caracteres, stop words y números.
    Para ello, tokeniza empleando un modelo de Spacy.
    """
    # Separar en tokens
    tokens = nlp(doc)
    # Filtrar tokens
    filtered_tokens = [t.lower_ for t in tokens if (len(t.text) >= nMinCharacters) and not t.is_punct and not re.match('[0-9]+', t.text)]
    # Recombinamos los tokens
    doc = ' '.join(filtered_tokens)
    
    return doc

def obtainSimilarity(doc, docIndex: list[str], vectorizerDict: dict[str]):
    """
    Obtiene la matriz de similitud calculada como la distancia coseno empleando los vectorizadores proporcionados.
    """
    similarityDict = {}
    for tag, vectorizer in vectorizerDict.items():
        vectorizedMatrix = vectorizer.fit_transform(doc)
        vectorizedDf = pd.DataFrame(vectorizedMatrix.toarray(), columns = vectorizer.get_feature_names_out(), index = docIndex)
        similarity = cosine_similarity(vectorizedDf)
        similarityDf = pd.DataFrame(similarity, index = vectorizedDf.index, columns = vectorizedDf.index)
        similarityDict[tag] = similarityDf

    return similarityDict


def findMostSimilar(similarityDict: dict[str, pd.DataFrame], docIndex: list[str], nMostSimilars: int = 1, classes: None | list[str] = None):
    """
    Encuentra el valor más similar para todas las matrices de similitud proporcionadas.
    """
    # Crear df de resultados
    if classes is not None:
        docIndex = list(filter(lambda x: not (x in classes), docIndex))
    mostSimilarDf = pd.DataFrame(index = docIndex, columns = list(similarityDict.keys()))

    for tag, similarity in similarityDict.items():
        
        mostSimilarOptions = []

        for index, row in similarity.iterrows():
            if classes is None:
                # Eliminar la similitud consigo mismo
                rowFiltered = row.drop(index)
            else:
                # Saltar las filas de clases
                if index in classes:
                    continue
                # Seleccionar la similitud con las clases
                rowFiltered = row[classes]
            
            # Buscar la máxima similitud
            mostSimilarIdxs  = rowFiltered.nlargest(nMostSimilars).index.tolist()
            opt  = ', '.join([f"{i} ({rowFiltered[i]:.2f})" for i in mostSimilarIdxs])
            mostSimilarOptions.append(opt)

        # Escribir los resultados
        mostSimilarDf[tag] = mostSimilarOptions
    
    return mostSimilarDf

# python -m spacy download es_core_news_lg
nlp = spacy.load("es_core_news_lg")

# Cargar los datos
df = pd.read_csv('data/Distritos_v3.csv')

# Aplicar preprocesamiento
corpus = df['Información'].tolist()
descriptions = [normalizeDoc(nlp, doc) for doc in corpus]
descriptionIndex = list(df['Zona'].values)

# Obtener similitud entre barrios
models = {'BoW': CountVectorizer(), 'TF-IDF': TfidfVectorizer(), 'TF-IDF N-gram(1,3)': TfidfVectorizer(ngram_range = (1, 3))}
similarityDict = obtainSimilarity(descriptions, descriptionIndex, models)
mostSimilarDf = findMostSimilar(similarityDict, descriptionIndex)
print(mostSimilarDf)

# Obtener similitud con clases
with open('data/tipologyDescription.json', 'r', encoding = 'utf-8') as f:
    tipology: dict = json.load(f)

classes = list(tipology.keys())
classDescriptions = [normalizeDoc(nlp, doc) for doc in list(tipology.values())]
descriptionsAndClasses = descriptions + classDescriptions
descriptionAndClassesIndex = descriptionIndex + classes

similarityDict = obtainSimilarity(descriptionsAndClasses, descriptionAndClassesIndex, models)
mostSimilarDf = findMostSimilar(similarityDict, descriptionAndClassesIndex, 2, classes)
print(mostSimilarDf)