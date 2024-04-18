
# coding: utf-8

# In[1]:


# Importamos las librerías necesarias
from bs4 import BeautifulSoup
import requests
import pandas as pd
import html5lib
import os
import time
# Función para extraer las características de cada inmueble
def getFeatures(Features):
    H = 'None'
    M=   'None'
    for feature in Features: 
        if feature.text.find('hab')!=-1:
            H = (feature.text) #número de habitaciones
        elif feature.text.find('m'):
            M = (feature.text) # metros cuadrados
    
    return H,M


# Función para obtener texto en caso de que exista
def GetText(element):
    if element:
        Text= element.getText()
    else:
        Text='None'
    
    return Text

# Función para obtener la url de la imagen principal del inmueble
def getImgUrl(url): 
    headers = {'User-Agent': 'Mozilla/5.0'}
    # Realizamos la petición a la web
    time.sleep(3)
    req = requests.get(url, headers=headers, timeout=30)


    # Pasamos el contenido HTML de la web a un objeto BeautifulSoup()
    html = BeautifulSoup(req.text, 'html5lib')
    imgurl= html.findAll('img')
    
    
    try:
        return imgurl[0]['src']

    except:
        return None
    
        

# Función para obtener los datos de cada inmueble
def GetDatasetFromUrl(html,cont):
    panda = pd.DataFrame(columns=['Title','description','price','hab','m2','timeago','phone','url','imgurl'])

    containers= html.findAll('div',{'class':'re-Card'})
    imgurl = ''
    
    for element in containers:
        price= GetText(element.find('span',{'class':'re-Card-price'}))
        title = GetText(element.find('h3',{'class':'re-Card-title'}))
        description = GetText(element.find('span',{'class':'re-Card-description'}))        
        hab,m2 =  getFeatures(element.findAll('span',{'class':'re-Card-feature'}))
        timeago = GetText(element.find('span',{'class':'re-Card-timeago'}))        
        phone = GetText(element.find('span',{'class':'sui-AtomButton-text'}))        
        url =  element.find('a',{'class':'re-Card-link'})['href']
        
        imgurl = getImgUrl('https://www.fotocasa.es/' + url)
        #print(cont)
        if imgurl!=None:
             with open(os.path.join('pictures' , str(cont) + '.jpg'), 'wb') as handle: # las imágenes se guardan en la carpeta 'pictures'
                    response = requests.get(imgurl, stream=True)

                    if not response.ok:
                         print (response)

                    for block in response.iter_content(1024):
                         if not block:
                             break
 
                         handle.write(block)

        # Se guarda todo en un panda
        panda.loc[cont] = [title,description,price,hab,m2,timeago,phone,url,imgurl]
        
        cont+=1
    
    return panda,cont


# Función para recorrer todas las páginas de la búsqueda
def getNumberOfPages (html):
    listpages= []
    Pages = html.findAll('a',{'class':'sui-LinkBasic sui-PaginationBasic-link'})
    for page in Pages:
        
        listpages.append((page.getText()))
    if listpages==[]:
        return 1
    else:
        return listpages[-2]


# In[3]:


# Definimos la URL. Esta url se puede cambiar para poder elegir la ciudad de interés
url = "https://www.fotocasa.es/es/comprar/viviendas/valencia-capital/ciutat-vella/l" 


# Establecemos un user-agent
headers = {'User-Agent': 'Mozilla/5.0'}

# Realizamos la petición a la web
req = requests.get(url, headers=headers,timeout=25)

# Pasamos el contenido HTML de la web a un objeto BeautifulSoup()
html = BeautifulSoup(req.text, 'html5lib')


#Creamos el directorio en el cwd para guardar las imágenes de los inmuebles
if  os.path.isdir('pictures')==False:  
    os.mkdir('pictures')
    
# Definimos el número de páginas
Number = int( getNumberOfPages (html))

# Creamos un data frame donde guardaremos los datos obtenidos
df = pd.DataFrame(columns=['Title','Description','Price','hab','m2','Timeago','Phone','url','imgurl'])


# Creamos una lista de pandas para ir agregando los resultados de cada página
listOfPandas = [None] * (Number+1 )
cont=0


for i in range(Number) : 
    print('Scraping from page ' + str(i+1) + '/' + str(Number))
    
    listOfPandas[i],cont = GetDatasetFromUrl(html,cont)

    urlN = url + '/'+str(i+1)
    req = requests.get(urlN, headers=headers)
    html = BeautifulSoup(req.text, "html.parser")

# Agregamos la lista de pandas en un solo panda     
Panda_final = pd.concat(listOfPandas[:])


# In[4]:


# Guardamos el resultado en un fichero csv
Panda_final.to_csv('Fotocasa_scraping.csv',encoding='utf-8')


# In[5]:


# Inspeccionar el resultado final
Resultado = pd.read_csv('Fotocasa_scraping.csv')
Resultado
