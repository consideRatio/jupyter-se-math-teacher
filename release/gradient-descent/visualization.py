import numpy as np

from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

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
    
    def run(self, f, df_dx=None, df_dy=None, gd_trails=[], show_2d=True, show_3d=True):
        """Runs a gradient descent visualization."""
        # 1D arrays
        x, y = (np.linspace(X_0, X_1, NX), np.linspace(Y_0, Y_1, NY))
        # 2D arrays - sparse
        xx, yy = np.meshgrid(x, y, sparse=True)
        zz = f(xx, yy)
        # 2D arrays - dense
        xxx, yyy = np.meshgrid(x,y)

        # ...gd markers
        gd_trails_2d, gd_trails_3d = [], []
        for gd_trail in gd_trails:
            gd_trails_2d.append(go.Scatter(x=gd_trail[:,0], y=gd_trail[:,1], mode='markers+lines', line=dict(color='black'), showlegend=False))
            gd_trails_3d.append(go.Scatter3d(x=gd_trail[:,0], y=gd_trail[:,1], z=gd_trail[:,2]+0.05, mode='markers+lines', line=dict(color='black'), showlegend=False, marker=dict(size=3)))

        if (show_2d):
            # ...contours
            contour = go.Contour(x=x, y=y, z=zz, showscale=False, colorscale=COLORS, ncontours=NCONTOURS)

            # ...contours with quivers
            if (df_dx and df_dy):
                fig_quiver = ff.create_quiver(xxx, yyy, df_dx(xx, yy), df_dy(xx, yy), scale=0.1, arrow_scale=.4, line=dict(width=1), showlegend=False)

                fig_quiver['data'].extend((contour, *gd_trails_2d))
                fig_quiver['layout'] = self.layout
                fig_quiver['layout']['title'] = "A 2D to 1D-function's contour plot with gradient vectors"

                py.iplot(fig_quiver, show_link=False)
            
            # ...contours without quivers
            else:
                fig_contour = go.Figure(data=[contour], layout=self.layout) 
                fig_contour['layout']['title'] = "A 2D to 1D-function's contour plot"

                py.iplot(fig_contour, show_link=False)

        if (show_3d):
            # ... 3d surface
            surf = go.Surface(x=x, y=y, z=zz, showscale=False, colorscale=COLORS)

            # 3D figure
            fig_3d = go.Figure(data=[surf, *gd_trails_3d], layout=self.layout)
            fig_3d['layout']['title'] = "A 2D to 1D-function surface plot"

            # Render
            py.iplot(fig_3d, show_link=False)