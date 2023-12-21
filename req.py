import requests

url = 'http://127.0.0.1:5000/recognize'
files = {'image': open('C:\\Users\\amine\\OneDrive\\Bureau\\MA2\\AI_Project\\IA\\IA_Backend\\test.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.text)