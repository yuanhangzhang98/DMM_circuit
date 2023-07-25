# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:43:57 2021

Bidirectional logic operations used for building DMMs
Each "logic variable" is a continuous variable between [-1, 1]

@author: Yuanhang Zhang
"""

import numpy as np
import torch
import torch.nn as nn


class OR:
    def __init__(self, input_idx, input_sign):
        super(OR, self).__init__()
        self.shape_in = input_idx.shape
        self.n_sat = self.shape_in[-1]
        self.input_idx = input_idx
        self.input_sign = input_sign

        batch = self.shape_in[0]
        n_clause = self.shape_in[1]
        n = torch.max(input_idx).item() + 1
        n_appearance = torch.zeros(batch, n, dtype=torch.int64)
        n_appearance.scatter_add_(1, input_idx.reshape(batch, -1),
                                  torch.ones_like(input_idx.reshape(batch, -1)))
        max_len = torch.max(n_appearance).item()
        idx = self.input_idx

        # dv_in_idx = torch.zeros(batch, n_clause, 3, dtype=torch.int64)
        # appearance_count = torch.zeros(batch, n, dtype=torch.int64)
        # for i in range(batch):
        #     for j in range(n_clause):
        #         for k in range(3):
        #             # manually flattening the last two dimensions
        #             # dv_in: (batch, n, max_len) -> (batch, n * max_len)
        #             dv_in_idx[i, j, k] = appearance_count[i, idx[i, j, k]] + idx[i, j, k] * max_len
        #             appearance_count[i, idx[i, j, k]] += 1

        # idx = idx.reshape(batch, n_clause * 3)
        # cumsum = torch.zeros(batch, n, n_clause * 3, dtype=torch.int64)
        # cumsum.scatter_(1, idx.unsqueeze(1), torch.ones_like(idx.unsqueeze(1)))
        # cumsum = cumsum.cumsum(2)
        # appearance_count = torch.gather(cumsum, 1, idx.unsqueeze(1)).reshape(batch, n_clause, 3) - 1
        # dv_in_idx = appearance_count + idx.reshape(batch, n_clause, 3) * max_len

        dv_in_idx = torch.zeros(batch, n_clause, 3, dtype=torch.int64)
        for i in range(batch):
            batch_idx = idx[i].reshape(-1)
            cumsum = torch.zeros(n, n_clause * 3, dtype=torch.int64)
            cumsum.scatter_(0, batch_idx.unsqueeze(0), torch.ones_like(batch_idx.unsqueeze(0)))
            cumsum = cumsum.cumsum(1)
            appearance_count = torch.gather(cumsum, 0, batch_idx.unsqueeze(0)).reshape(n_clause, 3) - 1
            dv_in_idx[i] = appearance_count + idx[i] * max_len

        # assert torch.all(torch.eq(dv_in_idx, dv_in_idx_1))

        self.dv_in_idx = dv_in_idx.reshape(batch, -1)
        self.batch = batch
        self.n = n
        self.n_clause = n_clause
        self.max_len = max_len
        
    @torch.no_grad()
    def init_memory(self, v):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2
        
        self.xl = torch.ones(self.shape_in[:-1])
        self.xs = C
    
    @torch.no_grad()
    def calc_grad(self, v, param):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign
        input = input.reshape(-1, self.n_sat)
        batch = input.shape[0]
        
        xs = self.xs.reshape(-1, 1)
        [alpha, beta, gamma, delta, zeta] = param
        epsilon = 1e-3

        v_top, v_top_idx = torch.topk(input, 1, dim=-1)
        v_top = (1 - v_top) / 2
        
        C = v_top[:, 0]
        G = C.unsqueeze(-1).repeat(1, self.n_sat)
        # G[torch.arange(batch), v_top_idx[:, 0]] = v_top[:, 1]
        G *= xs
        
        R = torch.zeros(batch, self.n_sat)
        R[torch.arange(batch), v_top_idx[:, 0]] = C
        R *= (1-xs)
        
        dv = (G + zeta*R).reshape(self.shape_in) * self.input_sign
        dv_R_intermediate = R.reshape(self.shape_in) * self.input_sign
        dxl = (alpha * (C-delta)).reshape(self.xl.shape)
        dxs = (beta * (xs.squeeze()+epsilon) * (C-gamma)).reshape(self.xs.shape)

        dv_R = torch.zeros(self.batch, self.n)
        dv_R.scatter_add_(1, self.input_idx.reshape(batch0, -1), dv_R_intermediate.reshape(batch0, -1))
        
        # v.grad.data.scatter_add_(1, self.input_idx.reshape(batch0, -1), dv.reshape(batch0, -1))
        dv_in = torch.zeros(self.batch, self.n * self.max_len)
        dv_in.scatter_(1, self.dv_in_idx, dv.reshape(self.batch, -1))
        dv_in = dv_in.reshape(self.batch, self.n, self.max_len)
        xl_in = -100 * torch.ones(self.batch, self.n * self.max_len)
        xl_in.scatter_(1, self.dv_in_idx, torch.log(self.xl).unsqueeze(-1)\
                       .expand(self.shape_in).reshape(self.batch, -1))
        xl_in = xl_in.reshape(self.batch, self.n, self.max_len)

        return dv_in, xl_in, dv_R, dxl, dxs, C.reshape(batch0, -1)
    
    @torch.no_grad()
    def calc_C(self, v):
        batch0 = v.shape[0]
        input = v[torch.arange(batch0).view((batch0, 1, 1)), \
                  self.input_idx]
        input = input * self.input_sign        
        C = torch.max(input, dim=-1)[0]
        C = (1 - C) / 2
        return C
    
    @torch.no_grad()
    def clamp(self, max_xl):
        self.xl.data.clamp_(1, max_xl)
        self.xs.data.clamp_(0, 1)
        