import numpy as np
class SlidingWindow:
    def __init__(self, text, window_length=512, start=0):
        self.text = text
        self.window_length = window_length
        self.start = start
    def sample(self):
        return self.text[self.start:self.start+self.window_length]
    def slide(self):
        self.start = (self.start + self.window_length) % len(self.text)
    def target(self):
        return self.text[self.start+1:self.start+self.window_length+1]