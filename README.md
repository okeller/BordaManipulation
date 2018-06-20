
# New Approximations for Coalitional Manipulation in General Scoring Rules
This repository contains the code for several works, dealing with approximations for the problem of coalitional manipulation under different scoring rules:

* Orgad Keller, Avinatan Hassidim, Noam Hazon:  **New Approximations for Coalitional Manipulation in Scoring Rules**, manuscript (*).
* Orgad Keller, Avinatan Hassidim, Noam Hazon:  **New Approximation for Borda Coalitional Manipulation**. AAMAS 2017: 606-614 (^). 
* Other work currently in progress.

## Requirements:
* Python 2.7 and up
* future
* NumPy
* SciPy
* cvxopt
* glpk
* Pandas
* mock (for tests)
* scikit-learn (only for joblib)

### Optional:
* Cython. In this case run
```commandline
python setup.py build_ext --inplace
```
## Running:

For **General-WCM**, using the algorithm in (*), follow this example:


```python
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import clp_R_alpha_WCM
import utils

# Creating the input:
initial_sigmas = np.array([10, 12, 12, 12, 14], dtype=np.int32)
alpha = utils.borda(5) * 2  # kind-of-Borda
weights = np.array([2, 1])
m = len(initial_sigmas)

# Running the algorithm:
fractional_makespan, clp_res = clp_R_alpha_WCM.find_strategy(initial_sigmas, alpha, weights, mode='per_cand')
clp_makespan = utils.weighted_makespan(clp_res, alpha, weights, initial_sigmas)
```

Final output should be in the form of:


```python
logger.info(
    'frac={} CLP={}'.format(fractional_makespan, clp_makespan))
```

    20XX-XX-XX XX:XX:XX,XXX : INFO : frac=26.000000011278484 CLP=26.0
    

For **General-UCM**, using the algorithm in (*), follow this example:


```python
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import clp_R_alpha_UCM
import utils

# Creating the input:
initial_sigmas = np.array([10, 12, 12, 12, 14], dtype=np.int32)
alpha = utils.borda(5) * 2  # kind-of-Borda
k = 2
m = len(initial_sigmas)

# Running the algorithm:
fractional_makespan, clp_res = clp_R_alpha_UCM.find_strategy(initial_sigmas, alpha, k, mode='per_cand')
clp_makespan = utils.makespan(initial_sigmas, clp_res, alpha=alpha)
```


```python
logger.info(
    'frac={} CLP={}'.format(fractional_makespan, clp_makespan))
```

    20XX-XX-XX XX:XX:XX,XXX : INFO : frac=20.000000083200916 CLP=20.0
    

For the specific case of **Borda-UCM**, using the algorithm in (^), follow this example:  


```python
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import clp_general
import utils

# Creating the input:
initial_sigmas = np.zeros(6, dtype=np.int32)
k = 3
m = len(initial_sigmas)

# Running the algorithm:
fractional_makespan, clp_res = clp_general.find_strategy(initial_sigmas, k, mode='per_cand')
clp_makespan = utils.makespan(initial_sigmas, clp_res)
```


```python
logger.info(
    'frac={} CLP={}'.format(fractional_makespan, clp_makespan))
```

    20XX-XX-XX XX:XX:XX,XXX : INFO : frac=7.500000196277758 CLP=8.0
    
