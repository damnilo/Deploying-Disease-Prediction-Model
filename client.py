import requests

data = {
  "symptoms": ["itching", "skin_rash", "dischromic _patches"]
}

url = 'http://localhost:8000/predict'

response = requests.post(url, json=data)

print(response.json())