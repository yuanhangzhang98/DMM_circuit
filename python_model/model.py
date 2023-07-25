# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 12:52:05 2021

@author: Yuanhang Zhang
"""


import numpy as np
import torch
import torch.nn as nn
from operators import OR
from dataset import SAT_dataset

class SAT:
    def __init__(self, batch, n, r):
        super(SAT, self).__init__()
        self.batch = batch
        self.n = n
        self.r = r
        self.max_xl = 1e4 * self.n
        # self.max_xl = 100

        self.dataset = SAT_dataset(batch, n, r)
        clause_idx, clause_sign = self.dataset.generate_instances()
        # clause_idx, clause_sign = self.dataset.import_data()
        self.OR = OR(clause_idx, clause_sign)
            
        self.v = 2 * torch.rand(batch, self.n) - 1
        self.v.grad = torch.zeros_like(self.v)
        
        self.OR.init_memory(self.v)

        self.state = State_module(self.v,
                                  [self.OR.xl],
                                  [self.OR.xs])

    def reset(self):
        self.v = 2 * torch.rand(self.batch, self.n) - 1
        self.OR.init_memory(self.v)
        self.state = State_module(self.v,
                                  [self.OR.xl],
                                  [self.OR.xs])


    def get_state(self):
        return self.state.detach()

    def backward(self, x, param):
        # x: State
        self.v = x.v
        self.OR.xl = x.xl[0]
        self.OR.xs = x.xs[0]
        dv = torch.zeros_like(self.v)
        dxl = []
        dxs = []
        is_solved = torch.ones(self.batch, dtype=torch.bool)

        param_i = [param['alpha'], param['beta'],
                   param[f'gamma'], param[f'delta'], param[f'zeta']]
        eta = param['eta']
        dv_in_i, xl_in_i, dv_R_i, dxli, dxsi, Ci = self.OR.calc_grad(self.v, param_i)
        weight = torch.softmax(xl_in_i, dim=2)  # (batch, n, max_len)
        dv = eta * torch.sum(dv_in_i * weight, dim=2) + dv_R_i  # (batch, n)
        dxl.append(dxli)
        dxs.append(dxsi)
        is_solved_i = (Ci < 0.5).all(dim=1)
        is_solved = is_solved & is_solved_i

        # max_dv = torch.max(dv.abs())
        # dt = (1 / max_dv).clamp(1e-3, 1)
        dt = 1.0
        return State(dv, dxl, dxs) * dt, dt, is_solved

class State_module(nn.Module):
    def __init__(self, v, xl, xs):
        super(State_module, self).__init__()
        self.v = nn.Parameter(v)
        self.xl = nn.ParameterList([nn.Parameter(xl_i) for xl_i in xl])
        self.xs = nn.ParameterList([nn.Parameter(xs_i) for xs_i in xs])

    def detach(self):
        return State(self.v.detach(),
                     [xl_i.detach() for xl_i in self.xl],
                     [xs_i.detach() for xs_i in self.xs])

    def set_grad(self, dx):
        self.v.grad = -dx.v
        for xl, dxl in zip(self.xl, dx.xl):
            xl.grad = -dxl
        for xs, dxs in zip(self.xs, dx.xs):
            xs.grad = -dxs

    def clamp_(self, max_xl=None):
        self.v.clamp_(-1, 1)
        for xl in self.xl:
            xl.clamp_(1, max_xl)
        for xs in self.xs:
            xs.clamp_(0, 1)

    @torch.no_grad()
    def add_noise(self, noise_strength):
        self.xl[0].data += noise_strength * torch.randn_like(self.xl[0])
        self.xs[0].data += noise_strength * torch.randn_like(self.xs[0])


class State:
    def __init__(self, v, xl, xs):
        self.v = v
        self.xl = xl
        self.xs = xs

    def __add__(self, other):
        return State(self.v + other.v,
                     [xl_1 + xl_2 for xl_1, xl_2 in zip(self.xl, other.xl)],
                     [xs_1 + xs_2 for xs_1, xs_2 in zip(self.xs, other.xs)])

    def __sub__(self, other):
        return State(self.v - other.v,
                     [xl_1 - xl_2 for xl_1, xl_2 in zip(self.xl, other.xl)],
                     [xs_1 - xs_2 for xs_1, xs_2 in zip(self.xs, other.xs)])

    def __mul__(self, other):
        # other is a python scalar
        return State(other * self.v,
                     [other * xl for xl in self.xl],
                     [other * xs for xs in self.xs])

    def __truediv__(self, other):
        # other is also a State. For computing relative error
        return State(self.v / other.v,
                     [xl_1 / xl_2 for xl_1, xl_2 in zip(self.xl, other.xl)],
                     [xs_1 / xs_2 for xs_1, xs_2 in zip(self.xs, other.xs)])

    def abs(self):
        return State(self.v.abs(),
                     [xl.abs() for xl in self.xl],
                     [xs.abs() for xs in self.xs])

    def clamp(self, max_xl=None):
        return State(self.v.clamp(-1, 1),
                     [xl.clamp(1, max_xl) for xl in self.xl],
                     [xs.clamp(0, 1) for xs in self.xs])

    def clamp_(self, max_xl=None):
        self.v.clamp_(-1, 1)
        for xl in self.xl:
            xl.clamp_(1, max_xl)
        for xs in self.xs:
            xs.clamp_(0, 1)

    def relative_error(self, x):
        max_error_list = [torch.max(self.v.abs())] \
                       + [torch.max(dxl.abs() / xl) for dxl, xl in zip(self.xl, x.xl)] \
                       + [torch.max(dxs.abs()) for dxs in self.xs]
        return max(max_error_list)