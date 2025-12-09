import math

class Value:
    def __init__(self, data, _parent_operands = (), _creator_op = '', label = ''):
        self.data = data
        self._grad = 0
        self._parent_operands = set(_parent_operands)
        self._creator_op = _creator_op
        self.label = label
        self.grad = 0 # the gradient of end with respect to this node
        self._backprop = lambda: None
        
    def __repr__(self):
        return f'Value(data = {self.data}, label = {self.label})'

    def __add__(self, operand2):
        operand2  = operand2 if isinstance(operand2, Value) else Value(data = operand2)
        out = Value(data = self.data + operand2.data, _parent_operands = (self, operand2), _creator_op = '+')
        #when a + b is called, it excutes a.__add__(b), a and b are both Value objs
        def _backward():
            self.grad += out.grad * 1
            operand2.grad += out.grad * 1
        out._backprop = _backward        
        return out

    def __radd__(self, operand2):
        return self + operand2

    def __sub__(self, operand2):
        out = self + (-1 * operand2)
        return out

    def __mul__(self, operand2):
        operand2  = operand2 if isinstance(operand2, Value) else Value(data = operand2)
        out = Value(data  = self.data * operand2.data, _parent_operands = (self, operand2), _creator_op = '*')
        #when a * b is called, it excutes a.__mul__(b), a and b are both Value objs
        def _backward():
            self.grad += out.grad * operand2.data
            operand2.grad += out.grad * self.data
        out._backprop = _backward
        return out

    def __rmul__(self, operand2):
        return self * operand2

    def __pow__(self, operand2):
        assert isinstance(operand2, (int, float)) # makes sure that the power is an int or float
        out = Value(data = self.data ** operand2, _parent_operands = (self, ), _creator_op = f'** {operand2}')
        def _backward():
            self.grad += out.grad * (operand2 * (self.data ** (operand2 - 1)))
        out._backprop = _backward
        return out

    def __truediv__(self, operand2): # self/other
        out = self * (operand2 ** -1)        
        return out
        
    def tanh(self):
        x = self.data
        tanhx = (1 - math.exp(-2*x))/(1 + math.exp(-2*x))
        out = Value(data = tanhx, _parent_operands = (self, ), _creator_op = 'tanh' )
        def _backward():            
            self.grad += out.grad * (1 - (tanhx ** 2))
        out._backprop = _backward       
        return out

    def exp(self):
        exp = math.exp(self.data)
        out = Value(data = exp, _parent_operands = (self, ), _creator_op = 'exp')
        def _backward():
            self.grad += out.grad * (out.data)
        out._backprop = _backward
        return out        

    def backProp(self):
        markedSet = set()
        ordering = []
        root = self
        def topoSort(root):
            if root in markedSet:
                return
            if not root._parent_operands:
                ordering.append(root)
                markedSet.add(root)
                return 
            for parent in root._parent_operands:
                topoSort(parent)
            markedSet.add(root)
            ordering.append(root)
            return
        topoSort(root)
        self.grad = 1        
        for node in ordering[::-1]:
            node._backprop()
        
        