#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:31:34 2018

@author: sviolante
"""

from sklearn.datasets import make_classification
from glmnet import LogitNet
X, y = make_classification(random_state=42)
m = LogitNet()
%time m = m.fit(X, y)

import glmnet, sklearn

print(glmnet.__version__, sklearn.__version__)

