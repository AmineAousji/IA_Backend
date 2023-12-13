import requests

url = 'http://127.0.0.1:5000/recognize'
files = {'image': open('C:\\Users\\amine\\Face-Recognition-master\\test.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.text)