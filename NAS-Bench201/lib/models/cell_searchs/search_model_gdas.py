###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import torch
import torch.nn as nn
from copy import deepcopy
from ..cell_operations import ResNetBasicblock
from .search_cells     import SearchCell
from .genotypes        import Structure


class TinyNetworkGDAS(nn.Module):

  #def __init__(self, C, N, max_nodes, num_classes, search_space, affine=False, track_running_stats=True):
  def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
    super(TinyNetworkGDAS, self).__init__()
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
        cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
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
    #self.hardwts_ = nn.Parameter( 1e-3*torch.zeros(num_edge, len(search_space)) )
    self.tau        = 10
   # self.arch_cache = self.genotype()

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

 # def hardwts(self):
 #   return self.hardwts_


  def set_tau(self, tau):
    self.tau = tau

  def get_tau(self):
    return self.tau

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

  def set_genotype(self, index):
    genotypes = []
    operations=[]
    k=0
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        ind=index[k][0].item()
        op_name  = self.op_names[ind]
        k=k+1
        xlist.append((op_name, j))
        operations.append(op_name)
      genotypes.append( tuple(xlist) )
    arch = Structure( genotypes )
    self.arch_cache = arch


  def set_gdas_genotype(self,set_cache):
    while True:
      gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
      logits  = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
      probs   = nn.functional.softmax(logits, dim=1)
      index   = probs.max(-1, keepdim=True)[1]
      if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
        continue
      else: break       
    genotypes = []
    operations=[]
    k=0
    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        ind=index[k][0].item()
        op_name  = self.op_names[ind]
        k=k+1
        xlist.append((op_name, j))
        operations.append(op_name)
      genotypes.append( tuple(xlist) )
    arch = Structure( genotypes )
    if set_cache: self.arch_cache = arch
    return index,arch

  def get_index_hardwts(self):
    while True:
      gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
      logits  = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
      probs   = nn.functional.softmax(logits, dim=1)
      index   = probs.max(-1, keepdim=True)[1]
      one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
      hardwts = one_h - probs.detach() + probs
      if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
        continue
      else: break
    return index,hardwts


  def forward(self, inputs,g,index,hardwts):
    if g==0:
      while True:
        gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
        logits  = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
        probs   = nn.functional.softmax(logits, dim=1)
        index   = probs.max(-1, keepdim=True)[1]
        one_h   = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        hardwts = one_h - probs.detach() + probs
        if (torch.isinf(gumbels).any()) or (torch.isinf(probs).any()) or (torch.isnan(probs).any()):
          continue
        else: break
  
  #  logits  = (self.arch_parameters.log_softmax(dim=1))
  #  index   = logits.max(-1, keepdim=True)[1]
   # probs   = nn.functional.softmax(logits, dim=1)  
    #index   = self.hardwts_.max(-1, keepdim=True)[1]    
    #hardwts = self.hardwts_ - probs.detach() + probs   
    
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell.forward_gdas(feature, hardwts, index)
        #feature = cell.forward_dynamic(feature, self.arch_cache)
      else:
        feature = cell(feature)
    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits,index,hardwts
