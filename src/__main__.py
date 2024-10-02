import argparse
from genthing import Pipeline, live_display, batch
from diffusers.utils import load_image
from consts import ZOOM, COUNT, DEVICE, DIMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--zoom', type=float, default=ZOOM)
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--device', default=DEVICE)

    args = parser.parse_args()

    image = load_image(args.filename).resize(DIMS)

    p = Pipeline(args.zoom, args.device)

    if args.live:
        live_display(p, image)
    else:
        batch(p, image, args.count)


if __name__ == "__main__":
    main()
