import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to download PDF files
def download_pdf(url, output_folder):
    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links on the webpage
        links = soup.find_all('a')
        
        # Iterate through all links to find PDF files
        for link in links:
            href = link.get('href')
            if href.endswith('.pdf'):
                # Construct the absolute URL of the PDF file
                pdf_url = urljoin(url, href)
                
                # Create the output folder if it doesn't exist
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                # Download the PDF file
                filename = os.path.join(output_folder, os.path.basename(pdf_url))
                with open(filename, 'wb') as f:
                    f.write(requests.get(pdf_url).content)
                
                print(f'Downloaded: {filename}')
    else:
        # If the request was not successful, print an error message
        print('Failed to retrieve the webpage. Status code:', response.status_code)

# URL of the webpage containing PDF links
url = 'https://example.com'

# Output folder to save downloaded PDF files
output_folder = 'pdf_files'

# Call the function to download PDF files from the webpage
download_pdf(url, output_folder)
