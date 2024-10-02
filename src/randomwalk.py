import random
from genthing import wrap, batch
from consts import POSITIVE, NEGATIVE


def random_walk_prompt():
    return wrap(random.choice([POSITIVE, NEGATIVE]), 0.25)


def random_walk(p, image, count):
    batch(p, image, count, prompt_hook=random_walk_prompt)
