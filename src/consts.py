import os
from dotenv import load_dotenv
load_dotenv()

ZOOM = 1.025
COUNT = 8
# this really can't get much lower as the output images become a blurly mess
# otherwise.
STRENGTH = 0.5

BASE = [
    'detailed', 'hyperdetailed', 'photorealistic', 'clear', 'Fujifilm XT3',
    'ISO 800', 'dslr', 'photo', 'realistic'
]

BASE = []

DIRECTION = []
EXTRA = DIRECTION + BASE

MUL = 0.75

POSITIVE = ['realistic', 'Fujifilm XT3', 'photorealistic', 'ISO 800', 'dslr']
NEGATIVE = ['anime', 'Studio Ghibli']

DEVICE = "mps"
DIMS = (512, 512)

EVERY = 10
REALIGN = 8

URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')

PROMPT_STRENGTH = 0.5
RANDOM_WALK_STRENGTH = 0.5
