import requests

response = requests.get("http://127.0.0.1:8000/ask", params={"q": "台灣在哪裡？"})
response.encoding = 'utf-8' 
print(response.json())