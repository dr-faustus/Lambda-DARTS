from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../../')
from sota.rnn.genotypes import STEPS
from sota.rnn.utils import mask2d, LockedDropout, embedded_dropout
from torch.autograd import Variable

INITRANGE = 0.04


class DARTSCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype
    self.x_mask = None
    self.h_mask = None
    

    # genotype is None when doing arch search
    steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE))
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)
    ])

  def forward(self, inputs, hidden, pert=None, redrop=True):
    T, B = inputs.size(0), inputs.size(1)

    if self.training:
      if redrop:
        x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
        h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
        self.x_mask =  deepcopy(x_mask)
        self.h_mask = deepcopy(h_mask)
      else:
        x_mask=self.x_mask
        h_mask=self.h_mask
    else:
      x_mask = h_mask = None

    hidden = hidden[0]
    hiddens = []
    for t in range(T):
      if pert is not None:
        hidden = self.cell(inputs[t], hidden, x_mask, h_mask,pert=pert)
      else:
        hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      # f = F.tanh
      f = torch.tanh
    elif name == 'relu':
      f = F.relu
    elif name == 'sigmoid':
      # f = F.sigmoid
      f = torch.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.recurrent):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast, 
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1,
                 cell_cls=DARTSCell, genotype=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        
        assert ninp == nhid == nhidlast
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype),cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx),cell_cls(ninp, nhid, dropouth, dropoutx)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls
        self.embdrop = None

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False, pert=None, redrop=False):
        batch_size = input.size(1)

        if redrop:
          emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
          self.embdrop  = emb.clone().detach()

        emb = self.lockdrop(self.embdrop, self.dropouti, redrop = redrop)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        c = 0

        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            if pert and self.cell_cls != DARTSCell:
              raw_output, new_h = rnn(current_input, hidden[0], pert=pert[c],redrop=False)
              c += 1
            else:
              raw_output, new_h = rnn(current_input, hidden[0])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            hidden = new_hidden

            output = self.lockdrop(raw_output, self.dropout, redrop=redrop)
            outputs.append(output)

            

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]

