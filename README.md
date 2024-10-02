# vibediffusion

A fun little experimental thing I wrote that generates little animations from
pictures with sdxl-turbo and blip based on the sentiment from a social media
feed.

blip captions the images, which is fed into the prompt for sdxl-turbo running in
img2img mode.

Much better things exist but I wrote this for fun.

Idea behind this was that I was curious about guiding image generation based on
something external, to try and match a vibe or get feedback from generation in
other ways than directly.
Using social media feeds (in this case mastodon) as that was the easiest to do
quickly.

## Usage

Tested on a M2 macbook, requires at least 16GB of RAM for the models.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run with the following to see it live:
```
python3 src live ./path/to/image
```

Or the following to save a gif:
```
python3 src batch ./path/to/image
```

The social media feed stuff is ran by:
```
export URL=mastodon-feed
export TOKEN=your-instance-auth-token
python3 src emotion ./path/to/image
```

There are a bunch of settings in `src/consts.py` that you will want to change.

## License

MIT
