
import numpy as np
from math import isclose
from IPython.display import display, Markdown
from visualization import Vis

class Sim:
    """A gradient descent simulation"""
    def __init__(self):
        self.vis = Vis()

        self.f = None
        self.df_dx, self.df_dy = None, None
        self.gradient_descent_step = None
        
        self.gd_trails = []

    
    def test_passed(self):
        #print("PASS - Your code passed the tests!")
        display(Markdown('![](img\pass.png)'))


    def test_failed(self, e):
        #print("FAIL - Your code failed the tests!")
        display(Markdown('![](img\\fail.png)'))

    
    # Simulation setup
    def setup_f(self, f):
        """Pass the function specified in the instructions (a function that we will investigate using GD)."""
        self.f = f

        # Test cases
        try:
            actual_data = [f(4,4), f(0,0), f(0,5), f(5,0), f(5,5)]
            correct_data = [0.0, 0.53455254198027291, -0.24808622997157845, -0.85934218863936607, 0.4794146557024872]
            assert np.allclose(actual_data, correct_data, atol=0)
            self.test_passed()
        except AssertionError as e:
            self.test_failed(e)


    def setup_grad_f(self, df_dx, df_dy):
        """Pass the two functions specified in the instructions (representing the numerical derivatives of the function f)."""
        self.df_dx = df_dx
        self.df_dy = df_dy

        # Test cases
        try:
            actual_data = [
                df_dx(4,4), df_dx(0,0), df_dx(0,5), df_dx(5,0), df_dx(5,5),
                df_dy(4,4), df_dy(0,0), df_dy(0,5), df_dy(5,0), df_dy(5,5)
            ]
            correct_data = [
                -4.57865284353384e-09, -0.67528633233632229, 1.1072220512945845, -0.4642336087448129, 0.435550702873988,
                4.6625163358499513e-09, 1.8222564213395964, -0.92753851759706796, 1.3567830224835431, 0.44203310716930955
            ]
            assert np.allclose(actual_data, correct_data, atol=0)
            self.test_passed()
        except AssertionError as e:
            self.test_failed(e)

    
    def setup_gds(self, gds):
        """Pass a function specified in the instructions that makes a gradient descent step. It should take the parameters x, y and return the values new_x, new_y and step_length."""
        self.gradient_descent_step = gds

        try:
            actual_data = [*gds(4,4, alpha=0.2), *gds(0,0, alpha=0.2), *gds(0,5, alpha=0.2), *gds(5,0, alpha=0.2), *gds(5,5, alpha=0.2)]
            correct_data = [4.0000000009157306, 3.9999999990674966, 1.3069526165403576e-09, 0.13505726646726446, -0.36445128426791928, 0.38867107408468843, -0.22144441025891692, 5.1855077035194137, 0.28887840850428093, 5.0928467217489626, -0.27135660449670862, 0.28680118644021058, 4.9128898594252028, 4.9115933785661383, 0.12411247843916039]
            assert np.allclose(actual_data, correct_data, atol=0)
            self.test_passed()
        except AssertionError as e:
            self.test_failed(e)

    def run(self, show_2d=True, show_3d=True):
        """Runs the simulation."""
        self.vis.run(self.f, self.df_dx, self.df_dy, self.gd_trails, show_2d=show_2d, show_3d=show_3d)