#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

class Logger(object):
    def __init__(self, fn):
        self.f = open(fn, 'w')

    def log(self, msg, *args, **kwargs):
        msg = msg.format(*args, **kwargs)
        print(msg)
        self.f.write(msg+"\n")
