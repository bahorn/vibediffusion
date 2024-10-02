import time
from diffusers import AutoPipelineForImage2Image
from transformers import pipeline as tpipeline
import matplotlib.pyplot as plt
import torch
from PIL import ImageEnhance, Image
from consts import DEVICE, EXTRA, DIMS, STEPS


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


class Pipeline:
    def __init__(self, zoom, device=DEVICE):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipe.safety_checker = \
            lambda images, clip_input: (images, [False] * len(images))
        pipe = pipe.to(device)
        self._pipe = pipe

        self._caption = tpipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device
        )

        # random parameters
        self._steps = STEPS
        self._zoom = zoom
        self._sharpness = 1.0005
        self._contrast = 1.0005
        self._color = 1.0005

    def step(self, image, addition=[]):
        steps = self._steps
        strength = 1 / steps

        prompt = self._caption(image)[0]['generated_text']
        prompt = [prompt + ', ' + ', '.join(addition + EXTRA)]
        print(prompt)

        if self._zoom != 1.0:
            curr = zoom_at(image, int(DIMS[0]/2), int(DIMS[1]/2), self._zoom)
        else:
            curr = image

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


class LiveDisplay:
    def __init__(self, p, image):
        self._fig, ax = plt.subplots()
        self._image = image
        self._im = ax.imshow(self._image)
        plt.show(block=False)
        self._fig.canvas.draw()
        plt.pause(0.01)
        self._p = p

    def step(self, prompt=[]):
        self._image = self._p.step(self._image, prompt)

        self._im.set_data(self._image)
        self._fig.canvas.draw()
        plt.pause(0.01)
        time.sleep(0.1)


def live_display(p, image):
    ld = LiveDisplay(p, image)
    while True:
        ld.step()


def batch(p, image, count):
    images = [image]

    for i in range(count):
        image = p.step(image)
        images.append(image)

    images[0].save(
        'out.gif',
        save_all=True,
        append_images=images[1::],
        duration=100,
        loop=1
    )
