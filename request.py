import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'e1/do':2, 'e2/do':9, 'p1/do':3, 'p2/do':4, 'Nr':4, 'fu/fy':6})

print(r.json())
