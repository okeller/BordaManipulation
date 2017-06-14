from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import logging
import sys

import numpy as np
from cvxopt import solvers
from mock import mock
from mock.mock import MagicMock
from nose.tools.nontrivial import nottest


import average_fit


from  lp_solver import *
import utils

solvers.options['show_progress'] = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import time

current_milli_time = lambda: int(round(time.time() * 1000))


class TestLpSolver(unittest.TestCase):
    # @nottest
    def test_lp_solve(self):
        pass