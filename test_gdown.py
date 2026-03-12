import os
import requests
import gdown

file_id = "17YMBYdqVsIy9tIhjMGpVuhtyrPTGVZXN"
url = f"https://drive.google.com/uc?id={file_id}"

print("Testing raw requests...")
try:
    response = requests.get(url, allow_redirects=True, stream=True)
    print(f"Status Code: {response.status_code}")
    print(f"Content Type: {response.headers.get('Content-Type')}")
    # Read a small chunk
    content = next(response.iter_content(chunk_size=1024))
    if b'html' in content[:100].lower():
        print("Response seems to be HTML (likely a virus warning or login page).")
    else:
        print("Response seems to be binary data.")
except Exception as e:
    print(f"Raw requests error: {e}")

print("\nTesting gdown fuzzy download...")
try:
    gdown.download(url=url, output="test_fuzzy.zip", quiet=False, fuzzy=True)
    print("Fuzzy download succeeded!")
except Exception as e:
    print(f"Fuzzy download error: {type(e).__name__}: {e}")
