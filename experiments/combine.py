"""
An idea, but didn't really work.
"""
import argparse
import os
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

OUT_DIM = 512

scale = int(OUT_DIM / int(WIDTH / math.sqrt(COUNT)))

SIZE = int(WIDTH / math.sqrt(COUNT))
GRID_SIZE = int(WIDTH / SIZE)

STRENGTH = 0.25
EXTRA = ", (anime)0.8, (studio ghibli)0.8, (green)0.8"


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


class Annotater:
    def __init__(self):
        self.captioner = tpipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
            device="mps"
        )

        self._description = None

    def annotate(self, frames):
        grid_image = create_image_grid(frames)
        if self._description is None:
            prompt = self.captioner(
                grid_image,
            )[0]['generated_text']
            self._description = prompt

        prompt = self._description

        # do an img2img step now.
        prompt = f'({prompt})2.0' + EXTRA
        print(prompt)

        return prompt, grid_image


class MockScale:
    def __init__(self):
        pass

    def upsample(self, image):
        return image


class LapSRNScale:
    def __init__(self, scale):
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = f"./models/LapSRN_x{scale}.pb"
        sr.readModel(path)
        sr.setModel("lapsrn", scale)

        sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        self._sr = sr

    def upsample(self, image):
        return self._sr.upsample(image)


class Batcher:
    def __init__(self, strength=STRENGTH, lora=None):
        self._strength = strength

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

        if lora:
            pipe.load_lora_weights(lora)

        pipe.safety_checker = \
            lambda images, clip_input: (images, [False] * len(images))

        self.pipe = pipe.to('mps')

    def step(self, prompt, image):
        # do an img2img step now.

        first = self.pipe(
            image=image,
            prompt=prompt,
            num_inference_steps=math.ceil(1 / self._strength),
            strength=self._strength,
            guidance_scale=0.0,

        ).images[0]

        first.save('changed.png')

        images = cut(first)

        return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--lora', default=None)
    parser.add_argument('--frames', default=0, type=int)
    parser.add_argument('--strength', default=STRENGTH, type=float)

    args = parser.parse_args()
    # Read a batch of frames from
    cap = cv2.VideoCapture(args.file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    images = []

    to_process = []

    frames = []

    count = 0

    annotater = Annotater()

    while cap.isOpened():
        if args.frames > 0 and count >= args.frames:
            break

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if len(frames) >= COUNT:
            to_process.append(annotater.annotate(frames))
            frames = []
        count += 1

    if len(frames) > 0:
        to_process.append(annotater.annotate(frames))

    frames = []

    print('applying filter')
    b = Batcher(strength=args.strength, lora=args.lora)
    for prompt, image in to_process:
        images += b.step(prompt, image)

    print('upscaling')

    sr = LapSRNScale(scale) if scale > 1.0 else MockScale()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (OUT_DIM, OUT_DIM))
    for image in tqdm(images):
        out.write(
            cv2.cvtColor(sr.upsample(to_numpy(image)), cv2.COLOR_RGB2BGR)
        )
    out.release()

    os.system('ffmpeg -y -i output.avi out.mp4')


if __name__ == "__main__":
    main()
