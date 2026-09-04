import math
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph

def visualise(d):
    global n,visual
    n = 1
    visual = Digraph("this creates the diagram")
    visual.attr(rankdir = 'LR')
    visualiser_background(d,n)
    visual.render("checking_image",format = 'jpg', view='True')

def visualiser_background(given_set,n):
    visual.node(str(n),f"{given_set.label} | {given_set.data:0.4f} | {given_set.grad:0.4f}",shape = 'box')
    if given_set._op != '':
        n+=1
        visual.node(str(n),given_set._op,shape = 'circle')
        visual.edge(str(n),str(n-1))
        obj = list(given_set._prev)
        n+=1
        visual.node(str(n),f"{obj[1].label} | {obj[1].data:0.4f} | {obj[1].grad:0.4f}",shape = 'box')
        visual.edge(str(n),str(n-1))
        n+=1
        visual.node(str(n),f"{obj[0].label} | {obj[0].data:0.4f} | {obj[0].grad:0.4f}",shape = 'box')
        visual.edge(str(n),str(n-2))
        visualiser_background(obj[1],n)
        visualiser_background(obj[0],n)


class Value:
    def __init__ (self, data,_children=(),_op='',label=''):
        self.data = data
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0
    def __repr__ (self):
        return f"value : {self.data}, children {self._prev}"
    def __add__(self, other):
        out = Value(self.data+other.data,(self,other),'+')
        return out
    def __mul__(self, other):
        out = Value(self.data*other.data,(self,other),'*')
        return out
    def __sub__(self, other):
        out = Value(self.data-other.data,(self,other),'-')
        return out
    def __truediv__(self, other):
        out = Value(self.data/other.data,(self,other),'/')
        return out

a = Value(2.00,label='a')
b = Value(-3.00,label='b')
c = Value(10.00,label='c')
e = a*b ; e.label = 'e'
d = e + c ; d.label = 'd'
f = Value(-2.00,label='f')
L = d*f; L.label = 'L'

#BACK PROPAGATION

L.grad = 1.00
f.grad = 4.00
d.grad = -2.00
c.grad = -2.00
e.grad = -2.00
b.grad = -4.00
a.grad = 6.00

a.data += 0.1*a.grad
b.data += 0.1*b.grad
c.data += 0.1*c.grad
f.data += 0.1*f.grad
e = a*b ; e.label = 'e'
d = e + c ; d.label = 'd'
L = d*f; L.label = 'L'

visualise(L)



