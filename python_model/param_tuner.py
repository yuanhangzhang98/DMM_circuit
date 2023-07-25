# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:14:46 2021

@author: Yuanhang Zhang
"""

from model import SAT
from optimizer import Optimizer

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class ParamTuner():
    def __init__(self, n, r, attempt_id, max_step=int(1e5)):
        self.batch = 100
        self.n = n
        self.r = r
        self.max_step = max_step
        self.id = attempt_id
        self.dmm = SAT(self.batch, self.n, self.r)
        self.load_space()

    def load_space(self):
        self.space = {
            'zeta': hp.loguniform('zeta', np.log(1e-4), np.log(1e-1)),
            # 'zeta': hp.uniform('zeta', 2.9e-3, 3.1e-3),
            # 'eta': hp.loguniform('eta', np.log(50), np.log(5000)),
            # 'lr': hp.loguniform('lr', np.log(0.01), np.log(1))
        }

    def objective(self, param):
        self.dmm.reset()
        optim = Optimizer(self.dmm, param, max_step=self.max_step)
        step, unsolved = optim.solve()
        # step = torch.quantile(step, 0.5).detach().cpu().numpy()
        return {
            'loss': unsolved,
            'param': param,
            'status': STATUS_OK
        }

    def optimize(self):
        max_evals = 100
        tpe_trials = Trials()
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest,
                    max_evals=max_evals, trials=tpe_trials)
        return tpe_trials, best


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor
                                      if torch.cuda.is_available()
                                      else torch.FloatTensor)
    try:
        os.mkdir('results/')
    except:
        pass
    ns = [10, 20, 50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000]
    r = 4.3
    n_attempts = 1
    # max_step = int(5e3)
    func = lambda x: int(np.exp(-2.65) * x ** 2.12)
    # for attempt_id in range(n_attempts):
    for n in ns:
        max_step = func(n)
        attempt_id = 0
        tuner = ParamTuner(n, r, attempt_id, max_step)
        tpe_trials, best = tuner.optimize()
        json.dump(tpe_trials.results, open('results/results_{}_{}.json'.format(tuner.n, attempt_id), 'w'))
        json.dump(best, open('results/best_param_{}_{}.json'.format(tuner.n, attempt_id), 'w'))
        print('Attempt {} completed.'.format(attempt_id))

