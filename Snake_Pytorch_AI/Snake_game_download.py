import requests
import os

dirpath = os.path.dirname(__file__)
file_path = os.path.join(dirpath,'snake_game.py')

if not os.path.exists(file_path): 
    with open(file_path,'wb') as f:
        request = requests.get('https://raw.githubusercontent.com/patrickloeber/python-fun/master/snake-pygame/snake_game.py')
        f.write(request.content)
else: 
    print('already downloaded')


