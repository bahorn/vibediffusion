"""
An idea, but didn't really work.
"""
import sys
import math
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderKL
from transformers import pipeline as tpipeline
import torch


WIDTH = 512
HEIGHT = 512

COUNT = 4

SIZE = int(512 / math.sqrt(COUNT))
GRID_SIZE = int(512 / SIZE)

EXTRA = ", (anime)2.0, (studio ghibli)2.0, (green)2.0"


def to_numpy(img):
    return np.asarray(img)


def to_pil(img):
    return Image.fromarray(np.uint8(img))


def create_image_grid(frames):
    images = [to_pil(frame).resize((SIZE, SIZE)) for frame in frames]

    # Determine grid size
    grid_size = int(math.sqrt(COUNT))

    # Find max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create canvas
    canvas_width = WIDTH
    canvas_height = HEIGHT
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')

    # Paste images
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        x = col * max_width
        y = row * max_height
        canvas.paste(img, (x, y))

    # Save result
    canvas.save('grid.png')
    return canvas


def crop_image(image, cell_width, cell_height, row, col):
    left = col * cell_width
    upper = row * cell_height
    right = left + cell_width
    lower = upper + cell_height
    return image.crop((left, upper, right, lower))


def cut(image):
    images = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = crop_image(image, SIZE, SIZE, row, col)
            images.append(cell)
    return images


def annotate(frames):
    # print(cv2.getBuildInformation())
    grid_image = create_image_grid(frames)

    # return "a large group of blue and white planets" + EXTRA, grid_image

    captioner = tpipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device="mps"
    )

    prompt = captioner(grid_image)[0]['generated_text']

    # do an img2img step now.
    prompt = prompt + EXTRA
    print(prompt)

    return prompt, grid_image


class Batcher:
    def __init__(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            add_watermarker=False
        )

        pipe.safety_checker = \
            lambda images, clip_input: (images, [False] * len(images))

        self.pipe = pipe.to('mps')

    def step(self, prompt, image):
        # do an img2img step now.
        strength = 0.5

        first = self.pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=math.ceil(1 / strength),
            strength=strength,
            guidance_scale=0.0,

        ).images[0]

        first.save('changed.png')

        images = cut(first)

        return images


def main():
    # Read a batch of frames from
    cap = cv2.VideoCapture(sys.argv[1])
    images = []

    to_process = []

    frames = []

    count = 0

    while cap.isOpened():
        if count >= 16:
            break

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) >= COUNT:
            to_process.append(annotate(frames))
            frames = []
        count += 1

    if len(frames) > 0:
        to_process.append(annotate(frames))

    frames = []

    print('applying filter')
    b = Batcher()
    for prompt, image in to_process:
        images += b.step(prompt, image)

    print('upscaling')

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)

    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

    res = []

    for image in tqdm(images):
        result = sr.upsample(to_numpy(image))
        res.append(to_pil(result))

    res[0].save(
        'out.gif',
        save_all=True,
        append_images=res[1::],
        duration=100,
        loop=1
    )


if __name__ == "__main__":
    main()
