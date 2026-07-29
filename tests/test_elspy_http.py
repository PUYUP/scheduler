import os
import requests

def download_elsevier_oa_pdf(doi: str, api_key: str, output_filename: str):
    """
    Downloads an open-access paper from Elsevier/ScienceDirect using its DOI.
    """
    # Clean the DOI in case it contains prefixes
    doi = doi.replace("DOI:", "").strip()
    
    # Construct the API endpoint URL for full-text article retrieval
    url = f"https://api.elsevier.com/content/article/doi/{doi}"
    
    # Configure headers required by Elsevier
    # We explicitly request application/pdf and view=FULL
    headers = {
        "X-ELS-APIKEY": api_key,
        "Accept": "application/pdf"
    }
    params = {
        "view": "FULL"
    }
    
    print(f"Requesting full text for DOI: {doi}...")
    
    try:
        # Send the GET request with stream enabled for file downloading
        response = requests.get(url, headers=headers, params=params, stream=True)
        
        # Check HTTP status code
        if response.status_code == 200:
            # Write the streaming content chunks directly into a PDF file
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"✅ Success! Paper saved as '{output_filename}'.")
            
        elif response.status_code == 403:
            print(f"❌ Error 403: Forbidden. The paper might not be Open Access, "
                  f"or your API key is invalid/unauthorized.")
        elif response.status_code == 404:
            print(f"❌ Error 404: Article not found. Check if the DOI is correct.")
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Execution Example ---
if __name__ == "__main__":
    # Replace this with your actual 32-character Elsevier API key
    MY_API_KEY = "17be9a73ed96f86fd7463575236a9285"
    
    # Example Open Access DOI (from an Elsevier Journal)
    # Ensure you use a known Open Access DOI for testing
    sample_doi = "10.1016/j.csi.2021.103565" 
    output_pdf = "elsevier_paper.pdf"
    
    download_elsevier_oa_pdf(sample_doi, MY_API_KEY, output_pdf)
