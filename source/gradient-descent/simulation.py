import numpy as np
from visualization import Vis

class Sim:
    """A gradient descent simulation"""
    def __init__(self):
        self.vis = Vis()

        # Default starting points
        self.starting_points = [
            (0.1, 2.75),
            (2.0, 2.0),
            (2.6, 2.6),
            (2.4, 0.2),
        ]

        self.gamma = 0.2
        self.precision = 0.001

        self.f = lambda x, y: np.sin((x/2-2)**2 + (y/2-2)**2) * np.cos(x-y + np.exp(-y))
        
        # Solutions
        self._h = 0.001
        self.dfx, self.dfy = self.dfx_solution, self.dfy_solution
        self.gradient_descent_step = self.gradient_descent_step_solution


    def dfx_solution(self, x, y):
        return (self.f(x+self._h,y) - self.f(x-self._h,y)) / (2*self._h)

    def dfy_solution(self, x, y):
        return (self.f(x,y+self._h) - self.f(x,y-self._h)) / (2*self._h)


    def gradient_descent_step_solution(self, x, y, gamma):
        """Meant to be defined by the user...
        returns the step size taken and the the new x"""
        new_x, new_y = x - gamma * self.dfx(x, y), y - gamma * self.dfy(x, y)
        step_size = np.linalg.norm((new_x - x, new_y - y))
        return new_x, new_y, step_size

    
    def set_f(self, f):
        """Set the function to gradient descent on."""
        self.f = f


    def set_df(self, dfx, dfy):
        """Set a function that numerically derivatives the function f."""
        self.dfx = dfx
        self.dfy = dfy

    
    def set_gds(self, gds):
        """Set a function that numerically derivatives the function f."""
        self.gradient_descent_step = gds

    
    def set_starting_points(self, *starting_points):
        self.starting_points = starting_points


    def run(self):
        """Runs the simulation."""

        gds = []
        for starting_point in self.starting_points:
            steps = 0
            step_size = np.inf
            x, y = starting_point
            x_steps, y_steps, z_steps = [x], [y], [self.f(x,y)]

            print('Running GD: from ({:.2}, {:.2}), gamma: {:2}, precision: {:.3} ...'.format(x, y, self.gamma, self.precision))
            while True:
                x, y, step_size = self.gradient_descent_step(x, y, self.gamma)
                x_steps.append(x)
                y_steps.append(y)
                z_steps.append(self.f(x,y))
                steps += 1

                if step_size < self.precision:
                    print('Arrived local minimum occurs at ({:.2}, {:.2}) after {} steps.'.format(x, y, len(z_steps)))        
                    break
                elif steps >= 100:
                    print('Failed to close in on a local minimum in less than 100 steps.')
                    break
            print()

            gds.append(np.column_stack((x_steps, y_steps, z_steps)))

        self.vis.run(self.f, gds)
