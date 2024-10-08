import argparse
from genthing import Pipeline, live_display, batch
from randomwalk import random_walk
from sentiment import live_emotion
from feedcontrol import live_feed
from diffusers.utils import load_image
from consts import ZOOM, COUNT, DEVICE, DIMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('filename')
    parser.add_argument('--zoom', type=float, default=ZOOM)
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--lora', default=None)

    args = parser.parse_args()

    image = load_image(args.filename).resize(DIMS)

    p = Pipeline(args.zoom, args.lora, args.device)

    match args.type:
        case 'live':
            live_display(p, image)
        case 'batch':
            batch(p, image, args.count)
        case 'emotion':
            live_emotion(p, image)
        case 'feed':
            live_feed(p, image)
        case 'random-walk':
            random_walk(p, image, args.count)
        case _:
            raise Exception('unknown command')


if __name__ == "__main__":
    main()
