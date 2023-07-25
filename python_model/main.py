# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:48:02 2021

@author: Yuanhang Zhang
"""

import os
import time
import numpy as np
import torch
from model import SAT
from optimizer import Optimizer

torch.set_default_tensor_type(torch.cuda.FloatTensor\
                              if torch.cuda.is_available()\
                                  else torch.FloatTensor)

# torch.set_default_tensor_type(torch.cuda.HalfTensor\
#                               if torch.cuda.is_available()\
#                                   else torch.HalfTensor)

# torch.set_default_tensor_type(torch.cuda.BFloat16Tensor\
#                               if torch.cuda.is_available()\
#                                   else torch.BFloat16Tensor)

try:
    os.mkdir('results/')
except FileExistsError:
    pass


batch = 100
# ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
# r = 7
ns = [10, 20, 50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000, 5000]
r = 4.3


param = {
    'alpha': 5,
    'beta': 20,
    'gamma': 0.25,
    'delta': 0.05,
    'epsilon': 1e-3,
    'zeta': 3e-3,
    'eta': 3000,
    'lr': 0.14
}

for n in ns:
    lr = 0.23 * n ** (-0.07)
    param['lr'] = lr
    zeta = np.exp(-6.5387503 + 6.83240998 * np.log(n) ** (-1.10258287))
    param['zeta'] = zeta
    # Initialize the model
    dmm = SAT(batch, n, r)
    optim = Optimizer(dmm, param, max_step=int(1e7))

    # Solve the instances and collect statistics
    t0 = time.time()
    solved_step, unsolved = optim.solve(save_trajectory=False)
    t1 = time.time()
    wall_time = t1 - t0
    solved_step = solved_step.to(torch.float32).detach().cpu().numpy()
    np.save('results/solved_step_{}_{}.npy'.format(n, r), solved_step)
    median_step = np.median(solved_step)
    with open('results/log.txt', 'a') as f:
        f.write(f'{n}\t{median_step}\t{wall_time}\n')

