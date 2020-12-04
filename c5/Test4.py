from mxnet import nd
from mxnet.gluon import nn
# -*- coding: utf-8 -*-
class Test4:
    def generate(self):
        X = nd.arange(16).reshape((1, 1, 4, 4))
        pool2d = nn.MaxPool2D(3, padding=1, strides=1)
        print(pool2d(X))
        
t = Test4()
t.generate()