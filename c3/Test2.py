from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
# -*- coding: utf-8 -*-
class Test2:
    def generate(self):
        num_inputs = 2
        num_examples = 1000
        true_w = [2, -3.4]
        true_b = 4.2
        features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
        labels = true_w[0]*features[:,0] + true_w[1] * features[:,1] + true_b
        labels += nd.random.normal(scale=0.01, shape=labels.shape)
        t.set_figsize()
        plt.scatter(features[:,1].asnumpy(), labels.asnumpy(), 1);
        batch_size = 10
        for X,y in t.data_iter(batch_size, features, labels):
            print(X, y)
            break
        
    def use_svg_display(self):
        display.set_matplotlib_formats('svg')
        
    def set_figsize(self, figsize=(3.5, 2.5)):
        self.use_svg_display()
        plt.rcParams['figure.figsize'] = figsize
    # 本函数已保存在d2lzh包中方便以后使用
    def data_iter(self,batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, batch_size):
            j = nd.array(indices[i: min(i + batch_size, num_examples)])
            yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素
            
    def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
        return nd.dot(X, w) + b
    
    def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
t = Test2()
t.generate()