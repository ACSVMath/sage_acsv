r"""A SageMath package for Analytic Combinatorics in Several Variables"""

import importlib.metadata
__version__ = importlib.metadata.version(__name__)

from sage_acsv.asymptotics import *
from sage_acsv.kronecker import kronecker_representation
from sage_acsv.helpers import get_expansion_terms
from sage_acsv.settings import ACSVSettings