import numpy as np
from mayavi.mlab import *
import numpy as np
from mayavi import mlab
from matplotlib.cm import get_cmap  # for viridis


def test_surf():
    """Test surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    print(np.shape(x), np.shape(y), np.shape(f))
    s = surf(x, y, f)
    show()
    #cs = contour_surf(x, y, f, contour_z=0)
    return s


def surf_test2():

    def f(x, y):
        return np.sin(2 * x) * np.cos(2 * y)

    # data for the surface
    x = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, x)
    Z = f(X, Y)
    print(np.shape(X), np.shape(Y), np.shape(Z))

    # data for the scatter
    xx = 4 * np.random.rand(1000) - 2
    yy = 4 * np.random.rand(1000) - 2
    zz = f(xx, yy)

    fig = mlab.figure(bgcolor=(1, 1, 1))
    # note the transpose in surf due to different conventions compared to meshgrid
    su = mlab.surf(X.T, Y.T, Z.T)
    sc = mlab.points3d(xx, yy, zz, scale_factor=0.1, scale_mode='none',
                       opacity=1.0, resolution=20, color=(1, 0, 0))

    # manually set viridis for the surface
    cmap_name = 'viridis'
    cdat = np.array(get_cmap(cmap_name, 256).colors)
    cdat = (cdat * 255).astype(int)
    su.module_manager.scalar_lut_manager.lut.table = cdat

    mlab.show()


if __name__ == "__main__":
    # test_surf()
    surf_test2()