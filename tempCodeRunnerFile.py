import math
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

class Value:
    def __init__ (self, data,_children=(),_op='',label=''):
        self.data = data
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda:None
    def __repr__ (self):
        return f"{self.label} : {self.data}"
    def __add__(self, other):
        if not isinstance(other,Value): other = Value(other)
        out = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad +=1.0*out.grad
        out._backward =_backward
        return out
    def __radd__(self,other):
        out = self+other
        return out
    def __mul__(self, other):
        if not isinstance(other,Value): other = Value(other,label=f"{other}")
        out = Value(self.data*other.data,(self,other),'*',f"{self.data:0.2f} * {other.data:0.2f}")
        def _backward():
            self.grad += other.data*out.grad
            other.grad +=self.data*out.grad
        out._backward =_backward
        return out
    def __rmul__(self,other):
        out = self*other
        return out
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1-out.data**2)*out.grad
        out._backward =_backward
        return out
    def exp(self):
        x = self.data
        t = math.exp(self.data)
        out = Value(t,(self,),'exp',f"exp ({self.label}) ")
        def _backward():
            self.grad += t*out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        assert isinstance(other,(int,float))
        t = self.data**other
        out = Value(t,(self,))
        def _backward():
            self.grad += (other*(self.data**(other-1)))*out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        out = self*(other**-1)
        out._op = '/'
        return out
    def __neg__(self):
        out = self*-1
        out.label = f'- ( {self.label} )'
        return out
    def __sub__(self,other):
        if not isinstance(other,Value): other = Value(other,label=f"{other}")
        out = self+(-other)
        out.label = f"{self.label} - {other.label}"
        return out
    def backpropagation(self):
        self.grad = 1.0
        obj = self
        topo = []
        visited = set()
        def _backward1(obj):
            if obj not in visited:
                visited.add(obj)
            for child in obj._prev:
                _backward1(child)
            topo.append(obj)
        _backward1(obj)
        for i in reversed(topo):
            i._backward()
    def visualise(self):
        def visualiser_sub(given):        
            if given._op != '' and given not in visited:
                visited.add(given)
                print(given)
                print(visited)
                visual.node(f"{n}{given._op}",f"{given._op}", shape = 'circle')
                visual.edge(f"{n}{given._op}",f"{n}")
                for child in given._prev:
                    n+=1
                    visual.node(f"{n}",f"{child.label} | {child.data:0.4f} | {child.grad:0.4f}", shape = 'box')
                    visual.edge(f"{n}",f"{n-1}{given._op}")
                visualiser_sub(child)
        visual = Digraph("this creates the diagram")
        visual.attr(rankdir = 'LR')
        visited = set()
        n = 1
        visual.node(f"{n}",f"{self.label} | {self.data:0.4f} | {self.grad:0.4f}", shape = 'box')   
        visualiser_sub(self)
        visual.render("checking_image",format = 'jpg', view='True')
        


# INPUTS
x1 = Value(2.00,label='x1')
x2 = Value(0.00,label='x2')
# WEIGHTS
w1 = Value(-3.00,label='w1')
w2 = Value(1.00,label='w2')
# BIAS

b = Value(6.881385,label='b')
# BIASED OUTPUT
x1w1 = x1*w1; x1w1.label='x1w1'
x2w2 = x2*w2; x2w2.label='x2w2'
x1w1x2w2 = x1w1 + x2w2 ; x1w1x2w2.label='x1w1 + x2w2'
n = x1w1x2w2 + b ; n.label = 'n'
#o = n.tanh(); o.label = 'o'
e = (x1).exp(); e.label = 'e'
e1 = e-1; e1.label = 'exp(2n) - 1'
e2 = e+1; e2.label = 'exp(2n) + 1'
o = (e1-e2) ; o.label = "o"
o.backpropagation()
o.visualise()