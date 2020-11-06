from mxnet import autograd, nd
class Test:
    
    """
    """

    def test(self):
        """
        Returns true if this set contains the specified element
        """
        x = nd.arange(4).reshape((4,1))
        x.attach_grad()
        print(autograd.is_training())
        with autograd.record():
            y = 2 * nd.dot(x.T, x)
            print(autograd.is_training())
        y.backward()
        assert(x.grad - 4*x).norm().asscalar() == 0
        print(x.grad)
        
    def f(self, a):
        b = a * 2
        while b.norm().asscalar() < 1000:
            b = b * 2
        if b.sum().asscalar() > 0:
            c = b
        else:
            c = 100 * b
        return c
    def test1(self):
        a = nd.random.normal(shape=1)
        a.attach_grad()
        with autograd.record():
            c = self.f(a)
        c.backward()
        assert(a.grad == c/a)
    def test2(self):
        print(dir(nd))
s = Test()
s.test2()              
                
                
                    
                    
                    
                    
                    
                    
                    