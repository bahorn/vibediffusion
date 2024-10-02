#!/usr/bin/env python
import json
import html2text
from websockets.sync.client import connect
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from genthing import LiveDisplay, Pipeline
from diffusers.utils import load_image
from consts import DIMS, ZOOM, DEVICE
import sys
import collections
import os

from dotenv import load_dotenv
load_dotenv()

URL = os.getenv('URL')
TOKEN = os.getenv('TOKEN')


h = html2text.HTML2Text()
h.ignore_links = True


class MessageRecv:
    def __init__(self, token=None):
        self._pipeline = create_analyzer(task="emotion", lang="es")
        if token:
            self._headers = {"Authorization": f"Bearer {token}"}
        self._batch = []

    def process_message(self, message):
        m = json.loads(message)

        if m['event'] != 'update':
            return False

        payload = json.loads(m['payload'])

        if payload['language'] != 'en':
            return False

        text = payload['content']

        if payload['sensitive']:
            text = payload['spoiler_text']

        text = preprocess_tweet(
            h.handle(text).strip().replace('\n', ' '),
            lang="en"
        )
        self._batch.append(text)

        return True

    def mood(self):
        batch = self._batch
        self._batch = []

        tags = []
        for msg in batch:
            tag = self._pipeline.predict(msg).output
            if tag != 'others':
                tags.append(tag)

        t = collections.Counter(tags).most_common(1)
        if len(t) == 0:
            return ['neutral expression']

        print(t)

        return [f'expressing {t[0][0]}']

    def loop(self, image):
        p = Pipeline(ZOOM, DEVICE)
        im = load_image(image).resize(DIMS)
        ld = LiveDisplay(p, im)
        with connect(URL, additional_headers=self._headers) as websocket:
            i = 0
            while True:
                message = websocket.recv()
                if self.process_message(message):
                    i += 1

                if (i % 25) == 0:
                    mood = self.mood()
                    ld.step(prompt=mood)


def main():
    mr = MessageRecv(token=TOKEN)
    mr.loop(sys.argv[1])


if __name__ == "__main__":
    main()
