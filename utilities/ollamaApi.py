import requests
import json
import pandas as pd

# Este script hace una llamada a un LLM en local
# Se puede usar con https://ollama.com/

# Configurar API
url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}

# Cargar datos
df = pd.read_csv('/../data/Distritos_v3.csv')
with open('/../data/tipologyDescription.json', 'r', encoding = 'utf-8') as f:
    tipology = json.load(f)

# Llamar al modelo
data = {
    "model": "llama3",
    "prompt":
            f'''You are an expert classifying neighborhoods from opinion texts in Spanish.
                You are going to be given a list that contains the name of the neighborhood and its corresponding opinion.
                You must answer with TWO classes for each description.
                These are the possible classes, structured in a json that contains the class as key and its description as value. Read them carefully:\n
                {tipology}.\n
                You can only use these classes and you must assign exactly two classes to each neighborhood.
                The answer must be formatted as a JSON with names as neighborhood keys and the list of two classes as values.
                Remember that the text is in Spanish.
                I do not want you to include any extra text or comment like "Here is the text" or "Here is the json output":\n
                {df[['Zona', 'Información']]}
            ''',
    "stream": False
}
response = requests.post(url, headers = headers, data = json.dumps(data))

# Obtener información de la respuesta
if response.status_code == 200:
    response_text = response.text
    data = json.loads(response_text)
    print(data["response"])
    try:
        actual_response = json.loads(data["response"])
    except:
        raise Exception('La respuesta contiene texto excedente o no está correctamente formateada como JSON.')
else:
    print("Error:", response.status_code, response.text)

# Añadir respuesta al df
df['clase'] = df['Zona'].map(actual_response)
print(df[['Zona', 'clase']])
