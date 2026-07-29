from atlazer.utils.gemini_batch import client
import os

from elsapy.elsclient import ElsClient
from elsapy.elsdoc import FullDoc


def main():
    config = {
        'apikey': '17be9a73ed96f86fd7463575236a9285',
        'insttoken': os.getenv('ELSEVIER_INST_TOKEN') or None,
        'url': 'https://api.elsevier.com/content/search/sciencedirect?query=star+trek+vs+star+wars',
        'origin': 'https://atlanize.com'
    }
    client = ElsClient(config['apikey'])
    client.inst_token = config['insttoken']

    try:
        doi_doc = FullDoc(doi='10.1016/0166-218X(92)90021-2')
        if doi_doc.read(client):
            print("doi_doc.title: ", doi_doc.title)
        else:
            print("Read document failed:", client.req_status)
    except Exception as e:
        print(f"Error reading document: {e}")
        print("Read document failed.")


if __name__ == '__main__':
    main()