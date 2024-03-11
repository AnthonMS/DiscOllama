import requests
import base64

def image_url_to_base64(url):
    response = requests.get(url)
    image_bytes = response.content
    base64_string = base64.b64encode(image_bytes).decode('utf-8')
    return base64_string