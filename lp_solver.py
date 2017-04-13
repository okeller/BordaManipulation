from cvxopt import solvers
from cvxopt.base import matrix
import numpy as np
import logging

logger = logging.getLogger(__name__)

OPTIMAL = 0
UNBOUNDED = 1
INFEASIBLE = 2

solver = 'glpk'

class HomogenicLpSolver(object):
    """
    Solves argmin cx subject to Ax<=0
    """

    def __init__(self, A, c, var_names=None, const_names=None):
        if var_names != None and len(var_names) != A.shape[1]:
            raise ValueError("len(var_names) != A.shape[1]")
        if const_names != None and len(const_names) != A.shape[0]:
            raise ValueError("len(const_names) != A.shape[0]")

        self.var_names = var_names
        self.const_names = const_names
        self.A = A
        self.b = np.zeros(A.shape[0], dtype=float)
        self.c = c


    def solve(self):
        c_ = matrix(self.c)

        # adding the dummy constraint
        A_ = np.vstack((self.A, -self.c))
        b_ = np.hstack((self.b, [1.0]))

        G = matrix(A_)
        h = matrix(b_)

        self.result = solvers.lp(c_, G, h, solver=solver)

        primal_objective_ = self.result['primal objective']
        status_ = self.result['status']
        if primal_objective_ < -0.5:
            logger.warning('cvxopt status={} with dummy'.format(status_))
            self.status = UNBOUNDED
        else:
            logger.warning('cvxopt status={}'.format(status_))
            self.status = OPTIMAL

        self.objective = primal_objective_

        _x = self.result['x']
        self._x = np.array(_x).flatten()

        _z = self.result['z']
        self._z = np.array(_z).flatten()[:-1]  # remove the dummy constraint

    @property
    def x(self):
        return self._x

    @property
    def z(self):
        return self._z

    def __getitem__(self, item):
        if self.var_names is not None and item in self.var_names:
            i = self.var_names.index(item)
            return self._x[i]
        elif self.const_names is not None and item in self.const_names:
            i = self.const_names.index(item)
            if i < len(self._z):
                return self._z[i]
            else:
                return None
        else:
            raise ValueError("{} can't be found in variables and constraints".format(item))


if __name__ == '__main__':
    A1 = -np.eye(5)
    b1 = np.zeros(5)
    c = np.ones(5)

    # x1+x2 >= 5
    A2 = np.array([[1, -1, 0, 0, 0]])
    b2 = np.array(0)

    A = np.vstack((A1, A2))
    b = np.hstack((b1, b2))

    var_names = ['v_{}'.format(i) for i in range(A.shape[1])]
    const_names = ['c_{}'.format(i) for i in range(A.shape[0])]
    solv = HomogenicLpSolver(A, c, var_names=var_names, const_names=const_names)
    solv.solve()

    print([(v, solv[v]) for v in var_names])
    print([(c, solv[c]) for c in const_names])
