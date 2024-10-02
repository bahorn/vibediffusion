import os
from dotenv import load_dotenv
load_dotenv()

ZOOM = 1.025
COUNT = 8
STEPS = 2


BASE = [
    'detailed', 'hyperdetailed', 'photorealistic', 'clear', 'Fujifilm XT3',
    'ISO 800', 'dslr', 'photo', 'realistic'
]

BASE = ["View of a village in the mountains"]

DIRECTION = []
EXTRA = DIRECTION + BASE

MUL = 0.75

POSITIVE = ['realistic', 'Fujifilm XT3', 'photorealistic', 'ISO 800', 'dslr'] # ['lush', 'green', 'healing']
NEGATIVE = ['anime', 'Studio Ghibli'] # ['fading', 'dark', 'rotting']

DEVICE = "mps"
DIMS = (512, 512)


EVERY = 10

URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')
