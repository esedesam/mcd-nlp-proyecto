import folium
import json
import numpy as np
import pandas as pd

# Load GeoJSON data (https://valencia.opendatasoft.com/explore/dataset/barris-barrios/export/)
with open('data/barris-barrios.geojson') as f:
    geoData = json.load(f)

# Load Color Mapping
with open('data/colorMappingBarrios.json') as f:
    colorMapping = json.load(f)

# Load Data to Show
dataDf = pd.read_csv('data/dummyData.csv', sep = ',')
barrioColName = 'barrio'
dataVarName = dataDf.columns[dataDf.columns != barrioColName]

# Add Data to GeoJSON
for feature in geoData['features']:
    idx = np.where(dataDf[barrioColName] == feature['properties']['nombre'])[0]
    if idx.size > 0:
        idx = idx[0]
        for varName in dataVarName:
            newValue = dataDf.loc[idx, varName]
            # Convert to Python Native Types for JavaScript Compatiblity
            if isinstance(newValue, np.int_):
                newValue = int(newValue)
            elif isinstance(newValue, np.generic):
                newValue = float(newValue)
            feature['properties'][varName] = int(dataDf.loc[idx, varName])
    else:
        feature['properties'][varName] = None
        raise UserWarning(f"No data found for {feature['properties']['nombre']}.")

# Create a base map
myMap = folium.Map(location = [39.475324399975754, -0.3525380336296802], zoom_start = 11)

# Add Barrios de Valencia to the map
folium.GeoJson(
    geoData,
    style_function = lambda feature: {
        'fillColor': colorMapping.get(feature['properties']['nombre'], 'gray'),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    },
    tooltip = folium.GeoJsonTooltip(fields = ['nombre'], labels = False),
    popup = folium.GeoJsonPopup(fields = dataVarName.tolist(), localize = False)
).add_to(myMap)

# Save the map to an HTML file
myMap.save("barris.html")