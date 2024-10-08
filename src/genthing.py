import time
import math
from compel import Compel, ReturnedEmbeddingsType
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
from transformers import pipeline as tpipeline
import matplotlib.pyplot as plt
import torch
from PIL import ImageEnhance, Image
from consts import DEVICE, EXTRA, DIMS, STRENGTH, REALIGN, PROMPT_STRENGTH


def wrap(prompts, weight):
    return list(map(lambda x: f'({x}){float(weight):.1f}', prompts))


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)


class Pipeline:
    def __init__(self, zoom, lora=None, device=DEVICE):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False
        )
        if lora:
            pipe.load_lora_weights(lora)

        pipe.safety_checker = \
            lambda images, clip_input: (images, [False] * len(images))
        pipe = pipe.to(device)
        self._pipe = pipe

        self._caption = tpipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device
        )

        self._prompter = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        # random parameters
        self._strength = STRENGTH
        self._zoom = zoom
        self._sharpness = 1.0005
        self._contrast = 1.0000
        self._color = 1.0005

        self._count = 0
        self._prompt_cache = []

    def step(self, image, addition=[]):
        strength = self._strength
        steps = math.ceil(1 / strength)

        if self._count % REALIGN == 0 or len(self._prompt_cache) == 0:
            self._prompt_cache.append(self._caption(image)[0]['generated_text'])
            self._prompt_cache = self._prompt_cache[-2:]

        if len(self._prompt_cache) > 1 and self._prompt_cache[-1] == self._prompt_cache[-2]:
            self._prompt_cache = self._prompt_cache[-1:]

        together = []
        if len(self._prompt_cache) == 1:
            together = wrap([self._prompt_cache[-1]], PROMPT_STRENGTH)
        elif len(self._prompt_cache) >= 2:
            if (1 - (self._count / REALIGN)) > 0:
                together += wrap(
                    [self._prompt_cache[-2]],
                    (1 - (self._count / REALIGN)) * PROMPT_STRENGTH
                )

            if (self._count / REALIGN) > 0:
                together += wrap(
                    [self._prompt_cache[-1]],
                    (self._count / REALIGN) * PROMPT_STRENGTH
                )

        prompt = ', '.join(together + addition + EXTRA)
        print(prompt)

        conditioning, pooled = self._prompter(prompt)

        if self._zoom != 1.0:
            curr = zoom_at(image, int(DIMS[0]/2), int(DIMS[1]/2), self._zoom)
        else:
            curr = image

        curr = ImageEnhance.Sharpness(curr).enhance(self._sharpness)
        curr = ImageEnhance.Contrast(curr).enhance(self._contrast)
        curr = ImageEnhance.Color(curr).enhance(self._color)

        self._count += 1
        self._count %= REALIGN

        return self._pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
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
        plt.pause(0.1)
        self._p = p

    def step(self, prompt=[]):
        self._image = self._p.step(self._image, prompt)

        self._im.set_data(self._image)
        self._fig.canvas.draw()
        plt.pause(0.1)
        time.sleep(0.1)


def live_display(p, image):
    ld = LiveDisplay(p, image)
    while True:
        ld.step()


def batch(p, image, count, prompt_hook=lambda: []):
    images = [image]

    for i in range(count):
        image = p.step(image, addition=prompt_hook())
        images.append(image)

    images[0].save(
        'out.gif',
        save_all=True,
        append_images=images[1::],
        duration=100,
        loop=1
    )
