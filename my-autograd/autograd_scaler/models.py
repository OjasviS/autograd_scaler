import random
from .value import Value

class Neuron:
    def __init__(self, inDimension):
        self.w = [Value(data = random.uniform(-1, 1)) for _ in range(inDimension)]
        self.b = Value(data = random.uniform(-1, 1))
        
    def __call__(self, x):
        # returns w.x + b
        wxi = [w * x for w, x in zip(self.w, x)]
        forwardPassRes = sum(wxi, self.b)        
        return forwardPassRes.tanh()
        
    def getParameters(self):
        return self.w + [self.b]
        
class Layer:
    def __init__(self, inDimension, nNeurons):
        self.layer = [Neuron(inDimension = inDimension) for _ in range(nNeurons)]
        
    def __call__(self, x):
        out = [n(x) for n in self.layer]
        return out[0] if len(out) == 1 else out

    def getParameters(self):
        param = []
        for i in self.layer:
            param.extend(i.getParameters())
        return param
        
class MLP:
    def __init__(self, inDimension, nLayerNeurons):
        layerDimAndSize = [inDimension] + nLayerNeurons
        self.layers = [Layer(layerDimAndSize[i], layerDimAndSize[i + 1]) for i in range(len(nLayerNeurons))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def getParameters(self):
        param = []
        for i in self.layers:
            param.extend(i.getParameters())
        return param
