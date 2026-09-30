from graphviz import Digraph
import math

class Value:
    def __init__(self,data,_child=(),_op="",label=""):
        self.data = data
        self._child = set(_child)
        self._op = _op
        self.label = label
        self.backward = lambda:None
        self.grad = 0.0
    def __repr__(self):
        out = f"{self.label} : {self.data}"
        return out
    def __add__(self,other):
        if not isinstance(other,Value): other = Value(other)
        out = Value(self.data+other.data,(self,other),"+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = backward
        return out
    def __radd__(self,other):
        out = self + other
        return out
    def __mul__(self,other):
        if not isinstance(other,Value): other = Value(other)
        out = Value(self.data*other.data,(self,other),"*")
        def backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out.backward = backward
        return out
    def __rmul__(self,other):
        out = self*other
        return out
    def __neg__(self):
        out = self * -1
        return out
    def __sub__(self,other):
        out = self + (-other)
        return out
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        t = self.data ** other
        out = Value(t,(self, ),f"{self.label}^{other}")
        def backward():
            self.grad += other*(self.data**(other-1))*out.grad
        out.backward= backward
        return out
    def __truediv__(self,other):
        out = self * (other**-1)
        return out
    def exp(self):
        t = math.exp(self.data)
        out = Value(t,(self,),"exp()")
        def backward():
            self.grad += out.data*out.grad
        out.backward = backward
        return out
    def visualise(self):
        node = set() ; edge  = set()
        def visualise_sub(obj):
          if obj not in node:
            node.add(obj)
            for child in obj._child:
              edge.add((child, obj))
              visualise_sub(child)
        visualise_sub(self)
        dot = Digraph(format = 'svg' , graph_attr = {'rankdir':'LR'})
        for n in node:
            print(n)
            dot.node(str(id(n)),f"{n.label} | {n.data:0.4f} | {n.grad:0.4f}", shape = "box")
            if n._op != "":
                dot.node(str(id(n)) + n._op , n._op, shape = "circle" )
                dot.edge(str(id(n)) + n._op , str(id(n)))
        for n1, n2 in edge:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot
    def _backward(self):
        topo = []
        visited = set()
        def topo_sort(obj):
            if obj not in visited:
                visited.add(obj)
                for child in obj._child:
                    topo_sort(child)
                topo.append(obj)
            return topo     
        topo = topo_sort(self)
        self.grad = 1
        for i in reversed(topo):
            i.backward()

x1 = Value(2.00) ; x1.label= "x1"
w1 = Value(-3.00) ; w1.label= "w1"
x2 = Value(0.00) ; x2.label= "x2"
w2 = Value(1.00) ; w2.label= "w2"
x1w1 = x1*w1 ; x1w1.label= "x1w1"
x2w2 = x2*w2 ; x2w2.label= "x2w2"
x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label= "x1w1 + x2w2"
b = Value(6.88137) ; b.label= "b"
n = x1w1x2w2 + b ; n.label = "n"
e = (2*n).exp() ; e.label= "e"
o = (e-1)/(e+1); o.label = "o"

o._backward()
dot = o.visualise()

# Render it to a file and open it automatically in your default image viewer/browser
dot.render('computation_graph', format='svg', view=True)
