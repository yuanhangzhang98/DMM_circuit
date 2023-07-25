# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:16:09 2021

@author: Yuanhang Zhang
"""

import numpy as np
import torch

class Optimizer():
    def __init__(self, dmm, param, max_step=int(1e6)):
        self.dmm = dmm
        self.batch = dmm.batch
        self.n = dmm.n
        self.param = {
            'alpha': 5,
            'beta': 20,
            'gamma': 0.25,
            'delta': 0.05,
            'epsilon': 1e-3,
            'zeta': 1e-3,
            'eta': 3000,
            'lr': 0.1
        }
        lr = 0.23 * self.n ** (-0.07)
        param['lr'] = lr
        self.param.update(param)
        
        # param_fixed = {
        #     'epsilon': 1e-3,
        #     'lr': 1
        #     }
        # self.param.update(param_fixed)
        
        self.lr = self.param['lr']
        self.optimizer = torch.optim.SGD(self.dmm.state.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.dmm.parameters(), lr=self.lr)
        self.max_step = max_step
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, \
        #                     mode='min', factor=0.5, patience=500, verbose=True)
        self.backward_fn = torch.compile(lambda x: self.dmm.backward(x, self.param))
        # self.backward_fn = lambda x: self.dmm.backward(x, self.param)

    @torch.no_grad()
    def solve(self, save_trajectory=False):
        is_solved = torch.zeros(self.batch, dtype=torch.bool)
        is_solved_last = is_solved.clone()
        solved_step = self.max_step * torch.ones(self.batch)

        if save_trajectory:
            v_trajectory = []
        
        for step in range(self.max_step):
            x = self.dmm.get_state()
            self.optimizer.zero_grad()
            dx, dt, is_solved_i = self.backward_fn(x)
            self.dmm.state.set_grad(dx)
            self.optimizer.step()
            self.dmm.state.clamp_(self.dmm.max_xl)

            if save_trajectory:
                v_trajectory.append(self.dmm.state.v.data.detach().clone())

            is_solved = is_solved | is_solved_i
            solved_step[is_solved^is_solved_last] = step
            is_solved_last = is_solved.clone()
            n_solved = torch.sum(is_solved).detach().cpu().numpy()
            print('n:', self.n, 'step:', step, '  unsolved:', self.batch - n_solved)
            # self.scheduler.step((self.batch - n_solved) / self.batch)
            # if n_solved == self.batch:
            if n_solved > 0.5 * self.batch:
                break

        if save_trajectory:
            v_trajectory = torch.stack(v_trajectory, dim=-1)
            return solved_step, self.batch - n_solved, v_trajectory
        else:
            return solved_step, self.batch - n_solved
