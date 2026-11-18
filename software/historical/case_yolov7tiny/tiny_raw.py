"""YOLOv7-tiny oficial: tres tensores de cabeza antes de decodificación/NMS."""
import copy
import torch
class TinyRaw(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.layers=copy.deepcopy(net.model[:-1])
        self.head=copy.deepcopy(net.model[-1].m)
        self.sources=list(net.model[-1].f)
    def forward(self,x):
        saved=[]
        for layer in self.layers:
            if layer.f!=-1:
                x=saved[layer.f] if isinstance(layer.f,int) else [x if j==-1 else saved[j] for j in layer.f]
            x=layer(x);saved.append(x)
        return tuple(self.head[i](saved[j]) for i,j in enumerate(self.sources))
