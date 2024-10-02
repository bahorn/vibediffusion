#!/usr/bin/env python
import json
import html2text
from websockets.sync.client import connect
from pysentimiento import create_analyzer
from pysentimiento.preprocessing import preprocess_tweet
from genthing import LiveDisplay, wrap
from consts import EVERY, URL, TOKEN, POSITIVE, NEGATIVE, MUL
import collections

h = html2text.HTML2Text()
h.ignore_links = True


class MessageRecv:
    def __init__(self, image_pipeline, token=None):
        self._pipeline = create_analyzer(task="sentiment", lang="en")
        self._headers = {}
        if token:
            self._headers = {"Authorization": f"Bearer {token}"}
        self._batch = []
        self._p = image_pipeline

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
        spec = 0
        for msg in batch:
            tag = self._pipeline.predict(msg).output
            print(tag)
            match tag:
                case 'POS':
                    spec += 1
                case 'NEG':
                    spec -= 1

        if spec == 0:
            return []
        elif spec > 0:
            return wrap(POSITIVE, spec * MUL)
        else:
            return wrap(NEGATIVE, abs(spec * MUL))

    def loop(self, image):
        ld = LiveDisplay(self._p, image)
        with connect(URL, additional_headers=self._headers) as websocket:
            i = 0
            just_done = False
            while True:
                message = websocket.recv()
                if self.process_message(message):
                    i += 1
                    just_done = False

                if (i % EVERY) == 0 and not just_done:
                    mood = self.mood()
                    ld.step(prompt=mood)
                    just_done = True


def live_emotion(p, file):
    mr = MessageRecv(p, token=TOKEN)
    mr.loop(file)
