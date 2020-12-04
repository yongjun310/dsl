import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import random
# -*- coding: utf-8 -*-
class Test10:
    def generate(self):
        net = nn.Sequential()
        net.add(nn.Dense(256, activation='relu'),
                nn.Dense(10))
        net.initialize(init.Normal(sigma=0.01))
        batch_size = 256
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        
        loss = gloss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
        num_epochs = 5
        d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
                      None, trainer)
    def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
        d2l.set_figsize(figsize)
        d2l.plt.xlabel(x_label)
        d2l.plt.ylabel(y_label)
        d2l.plt.semilogy(x_vals, y_vals)
        if x2_vals and y2_vals:
            d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
            d2l.plt.legend(legend)
t = Test10()
t.generate()