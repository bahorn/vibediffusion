# vibediffusion

A fun little experimental thing I wrote that generates little animations from
pictures with sdxl-turbo and blip.

blip captions the images, which is fed into the prompt for sdxl-turbo running in
img2img mode.

Much better things exist but I wrote this for fun.

## Usage

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run with the following to see it live:
```
python3 src --live ./path/to/image
```

Or the following to save a gif:
```
python3 src ./path/to/image
```

## License

MIT
