#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

class Logger(object):
    def __init__(self, fn):
        # doing a line buffering
        self.f = open(fn, 'w', 1)

    def log(self, msg, *args, **kwargs):
        msg = msg.format(*args, **kwargs)
        print(msg)
        self.f.write(msg+"\n")
