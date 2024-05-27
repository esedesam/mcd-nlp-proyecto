import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, accuracy_score

def getGoogleSeet(spreadsheet_id, outFilePath):
  
  url = f'https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=csv'
  response = requests.get(url)

  if response.status_code == 200:
    with open(outFilePath, 'wb') as f:
      f.write(response.content)
      print(f'CSV file saved to: {outFilePath}')    
  else:
    raise UserWarning(f'Error downloading Google Sheet: {response.status_code}')

outDir = 'tmp/'
outName = 'encuesta.csv'
filePath = os.path.join(outDir, outName)
os.makedirs(outDir, exist_ok = True)

downloadResults = True
if downloadResults:
    getGoogleSeet('1jnO3A1i_-xWuV2XiPuP0J-iyiUQYYqm9XZNetWi6HNg', filePath)

df = pd.read_csv(filePath)

# Preprocesado
df['Marca temporal'] = pd.to_datetime(df['Marca temporal'], format = '%d/%m/%Y %H:%M:%S')
df.columns = df.columns.tolist()[0:2] + [col.split(': ')[1] for col in df.columns[2:]]

# Filtrar en tiempo y quitar columna
currentTime = datetime.now()
todayStart = datetime(currentTime.year, currentTime.month, currentTime.day)
cutoffTime = currentTime - timedelta(minutes = 0)
df = df[(df['Marca temporal'] >= todayStart) & (df['Marca temporal'] <= cutoffTime)]
df.drop(columns = ['Marca temporal'], inplace = True)

# Añadir resultados del modelo
modelResultsPath = 'data/distritosClasificacionDef.csv'
aux = pd.read_csv(modelResultsPath, sep = ';', index_col = 0)
auxRow = aux.iloc[0]
aux_data = {'Nombre': 'Model Groundtruth'}
for col in aux.columns:
    aux_data[col] = auxRow[col]

df = pd.concat([df, pd.DataFrame([aux_data])], ignore_index = True)

# Obtener aciertos
classification_scores = {}

for index, row in df.iterrows():
    nombre = row['Nombre']
    if nombre != 'Model Groundtruth':
        scores = {}
        for district, classification in row.items():
            if district != 'Nombre':
                model_classification = df.loc[df['Nombre'] == 'Model Groundtruth', district].iloc[0]
                score = 1 if classification == model_classification else 0
                scores[district] = score
        classification_scores[nombre] = scores

scores_df = pd.DataFrame(data = classification_scores).T

# Calcular accuracy
scores_df['accuracy'] = scores_df.sum(axis = 1) / (scores_df.shape[1] - 1)
scores_df.sort_values(by = 'accuracy', ascending = False, inplace = True)

# Representar ganadores
colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['skyblue'] * (len(scores_df) - 3)
ax = scores_df['accuracy'].plot(kind = 'bar', figsize = (12, 8), color = colors)
ax.set_ylabel('Accuracy')

plt.show()