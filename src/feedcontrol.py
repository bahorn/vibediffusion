#!/usr/bin/env python
import json
import html2text
from pysentimiento.preprocessing import preprocess_tweet
from websockets.sync.client import connect
from genthing import LiveDisplay, wrap
from consts import EVERY, URL, TOKEN
from sentiment import MessageRecv
import spacy
import random

h = html2text.HTML2Text()
h.ignore_links = True


class FeedMessageRecv(MessageRecv):
    def __init__(self, image_pipeline, token=None):
        self._headers = {}
        if token:
            self._headers = {"Authorization": f"Bearer {token}"}
        self._batch = []
        self._p = image_pipeline
        self._nlp = spacy.load("en_core_web_sm")

    def mood(self):
        batch = self._batch
        ents = set()
        for row in batch:
            n = self._nlp(row)
            for ent in n.ents:
                ents.add(str(ent))

        if len(ents) >= 10:
            return wrap(list(random.sample(list(ents), 10)), 2.0)
        else:
            return wrap(list(ents), 2.0)


def live_feed(p, file):
    mr = FeedMessageRecv(p, token=TOKEN)
    mr.loop(file)
