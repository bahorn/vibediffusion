import argparse
import time
from diffusers import AutoPipelineForImage2Image
from transformers import pipeline as tpipeline
import matplotlib.pyplot as plt
import torch
from diffusers.utils import load_image
from PIL import ImageEnhance, Image

COUNT = 8
STEPS = 4

EXTRA = [
    'sunset', 'getting dark', 'detailed', 'high detail', 'fine details',
    'hyperdetailed', 'photorealistic', 'zoomed in', 'clear', 'Fujifilm XT3',
    'ISO 800', 'dslr', 'photo', 'realistic'
]

DEVICE = "mps"

DIMS = (512, 512)


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


class Pipeline:
    def __init__(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe.safety_checker = \
            lambda images, clip_input: (images, [False] * len(images))
        pipe = pipe.to(DEVICE)
        self._pipe = pipe

        self._caption = tpipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
            device=DEVICE
        )

        # random parameters
        self._steps = STEPS
        self._zoom = 1.01
        self._sharpness = 1.0005
        self._contrast = 1.0005
        self._color = 1.0005

    def step(self, image):
        steps = self._steps
        strength = 1 / steps

        prompt = self._caption(image)[0]['generated_text']
        prompt = [prompt + ', ' + ', '.join(EXTRA)]
        print(prompt)

        curr = zoom_at(image, int(DIMS[0]/2), int(DIMS[1]/2), self._zoom)

        curr = ImageEnhance.Sharpness(curr).enhance(self._sharpness)
        curr = ImageEnhance.Contrast(curr).enhance(self._contrast)
        curr = ImageEnhance.Color(curr).enhance(self._color)

        return self._pipe(
            prompt=prompt,
            image=curr,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=0.0
        ).images[0]


def live_display(p, image):
    fig, ax = plt.subplots()
    im = ax.imshow(image)
    plt.show(block=False)
    fig.canvas.draw()
    plt.pause(0.01)

    while True:
        image = p.step(image)

        im.set_data(image)
        fig.canvas.draw()
        plt.pause(0.01)
        time.sleep(0.1)


def batch(p, image):
    images = [image]

    for i in range(COUNT):
        image = p.step(image)
        images.append(image)

    images[0].save(
        'out.gif',
        save_all=True,
        append_images=images[1::],
        duration=100,
        loop=1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--live', action='store_true')

    args = parser.parse_args()

    image = load_image(args.filename).resize(DIMS)

    p = Pipeline()

    if args.live:
        live_display(p, image)
    else:
        batch(p, image)


if __name__ == "__main__":
    main()
