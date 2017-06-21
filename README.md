
# New Approximation for Borda Coalitional Manipulation
This repository contains the code for the paper:

Orgad Keller, Avinatan Hassidim, Noam Hazon:  **New Approximation for Borda Coalitional Manipulation**, AAMAS 2017 606-614, as well as subsequent work currently in progress.

### Requirements:
* Python 2.7 and up
* future
* NumPy
* SciPy
* cvxopt
* glpk
* tqdm
* Pandas
* mock
* scikit-learn (only for joblib)

### Running:
For **Borda UCM**, follow this example:  


```python
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import clp_general
import utils


initial_sigmas = np.array([5, 6, 6, 6, 7], dtype=np.int32)
k = 2
m = len(initial_sigmas)

gaps = utils.sigmas_to_gaps(initial_sigmas, np.max(initial_sigmas))
logger.info('gaps={}'.format(gaps))

fractional_makespan, clp_res = clp_general.find_strategy(initial_sigmas, k, mode='per_cand')

clp_makespan = utils.makespan(initial_sigmas, clp_res)

logger.info(
    'k={} m={} frac={} CLP={}'.format(k, m, fractional_makespan, clp_makespan))
```

Final output should be in the form of:
```
2017-06-21 15:03:12,950 : INFO : k=2 m=5 frac=10.000000041600458 CLP=10.0
```
