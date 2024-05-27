import folium
import json
from folium.plugins import GroupedLayerControl, FloatImage
from folium.features import CustomIcon, DivIcon
import numpy as np
import pandas as pd
from branca.colormap import linear
import base64

# Load Barrios GeoJSON (https://valencia.opendatasoft.com/explore/dataset/barris-barrios/export/)
with open('/../data/barris-barrios.geojson') as f:
    geoBarrios = json.load(f)

# Load Distritos GeoJSON (https://valencia.opendatasoft.com/explore/dataset/districtes-distritos/export/)
with open('/../data/districtes-distritos.geojson') as f:
    geoDistritos = json.load(f)

# Load Color Mappings
with open('/../data/colorMappingBarrios.json') as f:
    colorMappingBarrios = json.load(f)
with open('/../data/colorMappingDistritos.json') as f:
    colorMappingDistritos = json.load(f)

# Load Icon Mapping
with open('icons/classIcons.json', encoding = 'utf-8') as f:
    iconMapper = json.load(f)

# Load Data to Show
dataDf = pd.read_csv('/../data/df_variables_importantes.csv', sep = ',')
distritoColName = 'Distrito'
dataVarName = dataDf.columns[dataDf.columns != distritoColName]

# Create Linear Colormap Based on Numeric Variable
numVarNameList = ['Extranjeros']

# Load Classification Results
distritosTC = pd.read_csv('/../data/classificationForMap.csv', sep = ';', index_col = 0)

# Add Data to GeoJSON
for feature in geoDistritos['features']:
    idx = np.where(dataDf[distritoColName].str.lower().values == feature['properties']['nombre'].lower())[0]
    if idx.size > 0:
        idx = idx[0]
        for varName in dataVarName:
            newValue = dataDf.loc[idx, varName]
            # Convert to Python Native Types for JavaScript Compatiblity
            if isinstance(newValue, np.int_):
                newValueStr = f'{int(newValue):d}'
            elif isinstance(newValue, np.generic):
                newValueStr = f'{float(newValue):.2f}'
            feature['properties'][varName] = newValueStr
    else:
        raise UserWarning(f"No data found for {feature['properties']['nombre']}.")

# Create a base map
myMap = folium.Map(
    location = [39.475324399975754, -0.3525380336296802],
    zoom_start = 11,
    control_scale = True)

# Add Barrios de Valencia to the map
groupBarrios = folium.FeatureGroup('Barrios de Valencia').add_to(myMap)
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
    popup = folium.GeoJsonPopup(fields = ['nombre'], localize = False)
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
    popup = folium.GeoJsonPopup(fields = ['nombre'] + dataVarName.tolist(), localize = False)
).add_to(groupDistritos)

# Add Color-Codified Numeric Variable
groupNumVar = {}
for numVarName in numVarNameList:
    # Colormap
    colormap = linear.YlGn_09.scale(
        dataDf[numVarName].min(), dataDf[numVarName].max())
    colormapDict = dataDf.set_index(distritoColName)[numVarName]
    colormapDict.index = colormapDict.index.str.lower()

    # Map Layer
    groupNumVar[numVarName] = folium.FeatureGroup(numVarName).add_to(myMap)
    folium.GeoJson(
        geoDistritos,
        style_function = lambda feature: {
            'fillColor': colormap(colormapDict[feature['properties']['nombre'].lower()]),
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
        # Extract Class
        tag = distritosTC[classificationName].loc[distritosTC.index.str.lower().values == feature['properties']['nombre'].lower()].values[0]
        folium.Marker(
            location = [feature['properties']['geo_point_2d']['lat'], feature['properties']['geo_point_2d']['lon']],
            icon = CustomIcon(f'icons/{iconMapper[tag]}', icon_size = (40, 40)),
            tooltip = tag
        ).add_to(groupTipoClass[classificationName])

# Add Layer Control
GroupedLayerControl(
    groups = {'Capas': [groupDistritos, groupBarrios] + list(groupNumVar.values()), 'Clasificaciones': list(groupTipoClass.values())},
    collapsed = True,
).add_to(myMap)

# Save the map to an HTML file
myMap.save('barris.html')