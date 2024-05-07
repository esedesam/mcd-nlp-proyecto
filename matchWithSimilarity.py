import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import spacy
import re
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

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
df = pd.read_csv('data/samu/Distritos_v3.csv')

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
with open('data/samu/tipologyDescription.json', 'r', encoding = 'utf-8') as f:
    tipology: dict = json.load(f)

classes = list(tipology.keys())
classDescriptions = [normalizeDoc(nlp, doc) for doc in list(tipology.values())]
descriptionsAndClasses = descriptions + classDescriptions
descriptionAndClassesIndex = descriptionIndex + classes

similarityDict = obtainSimilarity(descriptionsAndClasses, descriptionAndClassesIndex, models)
mostSimilarDf = findMostSimilar(similarityDict, descriptionAndClassesIndex, 1, classes)
print(mostSimilarDf)

# Obtener representación vectorial de lexemas y las palabras de los textos
words = []
for s in descriptions:
    word_list = s.split()
    words.extend(word_list)
lexemas = [nlp.vocab[orth] for orth in nlp.vocab.vectors]
lexemasRand = [t.text for t in np.random.choice(lexemas, 10000, replace = False)]
wordsForTsne = lexemasRand + words + classes
wordVectors = np.array([nlp(word).vector for word in wordsForTsne])

# Obtener embedding a partir de los vectores
tsne = TSNE(n_components = 2, random_state = 0, n_iter = 250, perplexity = 50, init = 'random', learning_rate = 'auto')
np.set_printoptions(suppress = True)
T = tsne.fit_transform(wordVectors)

# Representar lexemas
fig, ax = plt.subplots(figsize = (14, 8))
ax.scatter(T[:len(lexemasRand), 0], T[:len(lexemasRand), 1], c = 'steelblue', alpha = 0.1)

# Representar palabras con color por barrio
cmap = plt.get_cmap('viridis', len(descriptionIndex))

auxIdx = len(lexemasRand)
for i, description in enumerate(descriptions):
    # Filtrar por barrio
    nWords = len([word for word in description.split() if word])
    TFiltered = T[auxIdx:(auxIdx+nWords), :]
    auxIdx += nWords

    ax.plot(TFiltered[:, 0], TFiltered[:, 1], '.', c = cmap(i / len(descriptionIndex)))

# Añadir la leyenda de colores
colorTags = [(tag, cmap(i / len(descriptionIndex))) for i, tag in enumerate(descriptionIndex)]
colorLegend = [plt.Line2D([0], [0], marker = 'o', c = mcolors.to_rgb(color), label = tag) for tag, color in colorTags]
ax.legend(handles = colorLegend, loc = 'best')

# Representar clases
auxIdx = len(lexemasRand) + len(words)
ax.plot(T[auxIdx:, 0], T[auxIdx:, 1], 'x', c = 'red')

for label, x, y in zip(classes, T[auxIdx:, 0], T[auxIdx:, 1]):
    ax.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')

plt.show()