import folium
import json
import folium.plugins
from folium.features import DivIcon
import numpy as np
import pandas as pd
from branca.colormap import linear

# Load Barrios GeoJSON (https://valencia.opendatasoft.com/explore/dataset/barris-barrios/export/)
with open('data/barris-barrios.geojson') as f:
    geoBarrios = json.load(f)

# Load Distritos GeoJSON (https://valencia.opendatasoft.com/explore/dataset/districtes-distritos/export/)
with open('data/districtes-distritos.geojson') as f:
    geoDistritos = json.load(f)

# Load Color Mappings
with open('data/colorMappingBarrios.json') as f:
    colorMappingBarrios = json.load(f)
with open('data/colorMappingDistritos.json') as f:
    colorMappingDistritos = json.load(f)

# Load Data to Show
dataDf = pd.read_csv('data/dummyData.csv', sep = ',')
barrioColName = 'barrio'
dataVarName = dataDf.columns[dataDf.columns != barrioColName]

distritosTC = pd.read_csv('data/distritosClasificacion.csv', sep = ';', index_col = 0)

# Add Data to GeoJSON
for feature in geoBarrios['features']:
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
    
# Create Linear Colormap Based on Numeric Variable
numVarNameList = ['dummyValue']

# Create a base map
myMap = folium.Map(
    location = [39.475324399975754, -0.3525380336296802],
    zoom_start = 11,
    control_scale = True)

# Add Barrios de Valencia to the map
groupBarrios = folium.FeatureGroup('Barrios de Valencia').add_to(myMap)
popupField = ['nombre']
popupField.extend(dataVarName.tolist())
folium.GeoJson(
    geoBarrios,
    style_function = lambda feature: {
        'fillColor': colorMappingBarrios.get(feature['properties']['nombre'], 'gray'),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    },
    zoom_on_click = True,
    tooltip = folium.GeoJsonTooltip(fields = ['nombre'], labels = False),
    popup = folium.GeoJsonPopup(fields = popupField, localize = False)
).add_to(groupBarrios)

# Add Distritos de Valencia to the map
groupDistritos = folium.FeatureGroup('Distritos de Valencia').add_to(myMap)
folium.GeoJson(
    geoDistritos,
    style_function = lambda feature: {
        'fillColor': colorMappingDistritos.get(feature['properties']['nombre'], 'gray'),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5
    },
    zoom_on_click = True,
    tooltip = folium.GeoJsonTooltip(fields = ['nombre'], labels = False),
    popup = folium.GeoJsonPopup(fields = ['nombre'], localize = False)
).add_to(groupDistritos)

# Add Color-Codified Numeric Variable
groupNumVar = {}
for numVarName in numVarNameList:
    # Colormap
    colormap = linear.YlGn_09.scale(
        dataDf[numVarName].min(), dataDf[numVarName].max())
    colormapDict = dataDf.set_index(barrioColName)[numVarName]

    # Map Layer
    groupNumVar[numVarName] = folium.FeatureGroup(numVarName).add_to(myMap)
    folium.GeoJson(
        geoBarrios,
        style_function = lambda feature: {
            'fillColor': colormap(colormapDict[feature['properties']['nombre']]),
            'color': 'black',
            'weight': 1,
            'dashArray': '5, 5',
            'fillOpacity': 0.5,
        },
        tooltip = folium.GeoJsonTooltip(fields = ['nombre'], labels = False),
        popup = folium.GeoJsonPopup(fields = ['nombre', numVarName], localize = False)
    ).add_to(groupNumVar[numVarName])

    # Legend
    colormapLegend = colormap.to_step(n = 5)
    colormapLegend.caption = numVarName
    myMap.add_child(colormapLegend)

# Add Tipology Classification
groupTipoClass = {}
groupTipoClass['None'] = folium.FeatureGroup('None').add_to(myMap)
for classificationName in distritosTC.columns:
    groupTipoClass[classificationName] = folium.FeatureGroup(classificationName).add_to(myMap)
    for feature in geoDistritos['features']:
        # Extractt text
        tag = distritosTC[classificationName].loc[distritosTC.index.str.lower().values == feature['properties']['nombre'].lower()].values[0]
        folium.Marker(
            location = [feature['properties']['geo_point_2d']['lat'], feature['properties']['geo_point_2d']['lon']],
            icon = DivIcon(
                icon_size = (150, 36),
                icon_anchor = (7, 20),
                html = f'<div style="font-size: 8pt; color : black">{tag}</div>')
        ).add_to(groupTipoClass[classificationName])

# Add Layer Control
# folium.LayerControl().add_to(myMap)
folium.plugins.GroupedLayerControl(
    groups = {'Capas': [groupDistritos, groupBarrios] + list(groupNumVar.values()), 'Clasificaciones': list(groupTipoClass.values())},
    collapsed = True,
).add_to(myMap)

# Save the map to an HTML file
myMap.save('barris.html')