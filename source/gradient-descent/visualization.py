import numpy as np

from bqplot import pyplot as plt
import ipyvolume.pylab as p3

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff


"""

TODO: Let user enter plot coordinates some fancy way
    IDEA: gdplot drag scatter over contour map?

TODO: 

TODO: 

TODO: show plots in different tabs?

import ipywidgets as widgets

tab_contents = ['P0', 'P1']
children = [widgets.Text(description=name) for name in tab_contents]
tab = widgets.Tab()
tab.children = children
for i in range(len(children)):
    tab.set_title(i, str(i))
tab


"""



# Jupyter Notebook responsive sizes - Small: 618, Medium: 790, Large: 990
SMALL, MEDIUM, LARGE = (602, 790, 990)
NX, NY = (26, 26)
NCONTOURS = 10
X_0, X_1 = (0, 5)
Y_0, Y_1 = (0, 5)
COLORS = 'Picnic'

class Vis: 
    def __init__(self):
        # Initialization required by plotly
        py.init_notebook_mode(connected=True)

        self.layout = go.Layout(
            width=SMALL, height=SMALL,
            margin=dict(l=50, r=50, b=50, t=50),
            scene=dict(
                camera=dict(
                    up=dict(x=1, y=0, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.1, y=-1, z=2.25)
                )
            )
        )
    
    def run(self, f, df_dx, df_dy, gds):
        """Runs a gradient descent visualization."""
        # 1D arrays
        x, y = (np.linspace(X_0, X_1, NX), np.linspace(Y_0, Y_1, NY))
        # 2D arrays - sparse
        xx, yy = np.meshgrid(x, y, sparse=True)
        zz = f(xx, yy)
        # 2D arrays - dense
        xxx, yyy = np.meshgrid(x,y)


        """ bqplot CODE
        """

        """ ipyvolume CODE     
        
        from matplotlib import cm
        colormap = cm.bwr
        znorm = (zz - zz.min()) / zz.ptp()
        color = colormap(znorm)

        

        p3.figure()
        p3.style.use([])
        p3.plot_surface(xxx, yyy, zz, color=color[...,0:3])
        p3.show()   
        """


        # ...gd markers
        gd_steps_2d = []
        gd_steps_3d = []
        for gd in gds:
            gd_steps_2d.append(go.Scatter(x=gd[:,0], y=gd[:,1], mode='markers+lines', line=dict(color='black'), showlegend=False))
            gd_steps_3d.append(go.Scatter3d(x=gd[:,0], y=gd[:,1], z=gd[:,2]+0.05, mode='markers+lines', line=dict(color='black'), showlegend=False, marker=dict(size=3)))

        # ...contours
        contour = go.Contour(x=x, y=y, z=zz, showscale=False, colorscale=COLORS, ncontours=NCONTOURS)

        # ...quivers
        fig_quiver = ff.create_quiver(xxx, yyy, df_dx(xx, yy), df_dy(xx, yy), scale=0.1, arrow_scale=.4, line=dict(width=1), showlegend=False)

        # Contours/Quivers figure
        fig_quiver['data'].extend((contour, *gd_steps_2d))
        fig_quiver['layout'] = self.layout
        fig_quiver['layout']['title'] = "A 2D to 1D-function's contour plot"

        # ... 3d surface
        surf = go.Surface(x=x, y=y, z=zz, showscale=False, colorscale=COLORS)

        # 3D figure
        fig_3d = go.Figure(data=[surf, *gd_steps_3d], layout=self.layout)
        fig_3d['layout']['title'] = "A 2D to 1D-function in 3D"

        # Render
        py.iplot(fig_3d, show_link=False)
        py.iplot(fig_quiver, show_link=False)
        """ PLOTLY CODE

        """