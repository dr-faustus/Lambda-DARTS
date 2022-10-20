##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
########################################################
# DARTS: Differentiable Architecture Search, ICLR 2019 #
########################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import SearchCell
from .genotypes        import Structure
import math


class TinyNetworkDartsV1(nn.Module):

  def __init__(self, C, N, max_nodes, num_classes, search_space):
    super(TinyNetworkDartsV1, self).__init__()
    self._C        = C
    self._layerN   = N
    self.max_nodes = max_nodes
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev, num_edge, edge2index = C, None, None
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ResNetBasicblock(C_prev, C_curr, 2)
      else:
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space)
        if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
        else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(search_space)) )
    self.weights = []
    self.lambda_ = 0.0
    self.epsilon_0 = 0
    self.epsilon = 0
    self.gamma = 0

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def get_alphas(self):
    return [self.arch_parameters]

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def genotype(self):
    genotypes = []
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = self.arch_parameters[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def forward(self, inputs, pert=None):
    alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)

    feature = self.stem(inputs)
    self.weights = []
    idx = 0
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        w = alphas.clone()
        if self.training:
          w.retain_grad()
          self.weights.append(w)
        if pert is not None:
          w = w - pert[idx]
          idx += 1
        feature = cell(feature, w)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits

  def get_arch_grads(self):
    return [w.grad.data.clone().detach().reshape(-1) for w in self.weights]

  def get_corr(self, grads=None):
    if grads is None:
      grads = self.get_arch_grads()

    def corr(x):
      res = []
      norms = [x_.norm() for x_ in x]
      for i in range(len(x)):
        for j in range(i + 1, len(x)):
          res.append(
            (torch.dot(x[i], x[j]) / (norms[i] * norms[j])).item())
      return sum(res) / len(res)

    return corr(grads)

  def get_perturbations(self):
    layer_gradients = self.get_arch_grads()

    with torch.no_grad():
      weight = 1 / ((len(layer_gradients) * (len(layer_gradients) - 1)) / 2)
      u = [g / g.norm(p=2.0) for g in layer_gradients]
      sum_u = sum(u)
      I = torch.eye(sum_u.shape[0]).cuda()
      P = [(1 / g.norm(p=2.0)) * (I - torch.ger(u_l, u_l)) for g, u_l in zip(layer_gradients, u)]
      perturbations = [weight * (P_l @ sum_u).reshape(self.arch_parameters.shape) for P_l in P]

    self.epsilon = self.epsilon_0 / torch.cat(perturbations).norm(p=2.0).item()
    return [self.epsilon * p for p in perturbations]

  def get_full_grads(self, forward_grads, backward_grads):
    reg_grad = [(f - b).div_(2 * self.epsilon) for f, b in zip(forward_grads, backward_grads)]
    for param, g in zip(self.get_weights(), reg_grad):
      param.grad.data.add_(self.lambda_ * g)
