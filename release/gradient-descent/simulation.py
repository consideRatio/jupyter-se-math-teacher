
import numpy as np
from math import isclose
from visualization import Vis

class Sim:
    """A gradient descent simulation"""
    def __init__(self):
        self.vis = Vis()

        self.f = None
        self.df_dx, self.df_dy = None, None
        self.gradient_descent_step = None
        
        self.gd_trails = []
    
    # Simulation setup
    def setup_f(self, f):
        """Pass a function with parameters x and y, with a return value of z."""
        self.f = f

        # Test cases
        assert isclose(f(4,4), 0)

    def setup_grad_f(self, df_dx, df_dy):
        """Pass two functions representing the numerical derivatives of the function f."""
        self.df_dx = df_dx
        self.df_dy = df_dy

    
    def setup_gds(self, gds):
        """Pass a function that makes a gradient descent step. It should take the values x, y """
        self.gradient_descent_step = gds

    def run(self, show_2d=True, show_3d=True):
        """Runs the simulation."""
        self.vis.run(self.f, self.df_dx, self.df_dy, self.gd_trails, show_2d=show_2d, show_3d=show_3d)