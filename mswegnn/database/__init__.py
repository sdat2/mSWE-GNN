from .graph_creation import * # Ensures that all the modules have been loaded in their new locations *first*.
from . import graph_creation  # imports WrapperPackage/packageA
import sys
sys.modules['graph_creation'] = graph_creation  # creates a packageA entry in sys.modules