import pymysql
import datetime
import traceback
import sys
import time
from copy import deepcopy
from scipy import interpolate
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
from mayavi import mlab

def show_optimal_points(_db_params, _search_table, _exp_id, _day):
    con = pymysql.connect(host=_db_params["host"],
                          user=_db_params["user"],
                          password=_db_params["password"],
                          # db='experiment',
                          charset='utf8mb4',
                          cursorclass=pymysql.cursors.DictCursor)

    cur = con.cursor()
    cur.execute("use {}".format(_db_params["db"]))

    # in db we store at least two rows with same step_id
    # lets load them for every point in _todays_search_table and find mean Q value
    # also we will bubble search point with maximum of mean Q-value
    min_point = None

    for point in _search_table:

        comm_str = "select * from exp_data where date(end_time) = date('{}')" \
                   " and exp_id={} and step_id={};".format(
            _day, _exp_id, point['number']
        )
        print(comm_str)
        cur.execute(comm_str)
        rows = cur.fetchall()

        # lets get mean sum of q_val for that two rows
        q1 = rows[0]['q_val']
        q2 = rows[1]['q_val']

        mean_q = (q1 + q2) / 2

        f1 = rows[0]['f_val']
        f2 = rows[1]['f_val']

        mean_f = (f1 + f2) / 2

        # add that value to point as new key-value pair
        point.update({'mean_q': mean_q})
        point.update({'mean_f': mean_f})

        if not min_point:
            # if it is first iteration - set first point as min
            min_point = point
        else:
            # compare values of current point and max point
            if point['mean_q'] < min_point['mean_q']:
                min_point = point

    # lets create 3d plot with interpolation
    # our points not lies on grid, so we have to use griddata
    print("\n min point is : {} \n".format(min_point))
    print(_search_table)
    # prepare grid
    grid_x, grid_y = np.mgrid[10:250:1, 10:250:1]

    # prepare values on points array
    values = np.array([v['mean_q'] for v in _search_table])

    # prepare points array
    # [(x, y), ...]
    points = np.array([(v['red'], v['white']) for v in _search_table])

    grid_z0 = interpolate.griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z1 = interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z2 = interpolate.griddata(points, values, (grid_x, grid_y), method='cubic')

    print(np.shape(grid_z0))

    import matplotlib.pyplot as plt

    # plt.subplot(221)

    # plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')

    # plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)

    z = np.array([v['mean_q'] for v in _search_table])
    max_z = max([v['mean_q'] for v in _search_table])
    area = np.array([(v['mean_q']/max_z)*25 for v in _search_table])
    z_mark = np.array([v['mean_q']*1.5 for v in _search_table])
    x = np.array([v['red'] for v in _search_table])
    y = np.array([v['white'] for v in _search_table])

    # plt.title('Original')

    plt.subplot(222)

    plt.imshow(grid_z0.T)
    plt.scatter(x, y, s=area, c=z, antialiased=False, cmap=cm.coolwarm)

    plt.title('Nearest')

    plt.subplot(223)

    plt.imshow(grid_z1.T)
    plt.scatter(x, y, s=area, c=z, antialiased=False, cmap=cm.coolwarm)

    plt.title('Linear')

    plt.subplot(224)
    plt.ylabel('white')
    plt.xlabel('red')

    plt.imshow(grid_z2.T)
    plt.scatter(x, y, s=area, c=z, antialiased=False, cmap=cm.coolwarm)

    plt.title('Cubic')

    plt.gcf().set_size_inches(6, 6)

    plt.show()

    # now use interp2d
    # lets generate x, y ,z separately
    z = np.array([v['mean_q'] for v in _search_table])
    z_mark = np.array([v['mean_q']*1.5 for v in _search_table])
    x = np.array([v['red'] for v in _search_table])
    y = np.array([v['white'] for v in _search_table])
    for i in range(0, len(z)):
        print('x = {} y = {} z = {}'.format(x[i], y[i], z[i]))

    f = interpolate.interp2d(x, y, z, kind='linear')

    xnew = np.arange(10, 250, 1)
    ynew = np.arange(10, 250, 1)

    interp = f(xnew, ynew)
    #
    xx_new, yy_new = np.meshgrid(xnew, ynew, indexing='ij')


    fig = plt.figure()
    cs = plt.contour(xx_new, yy_new, interp.T, )
    # pl.plot(maxes[::, 1], maxes[::, 2], "-ob", label='trajectory of maximum')
    plt.clabel(cs, fmt='%.1f')  # , colors="black")
    fig.colorbar(cs, shrink=0.5, aspect=5)

    plt.ylabel('white')
    plt.xlabel('red')
    # ax.set_zlabel('dCO2/dt *-1')
    plt.title(_day)
    # fig.legend(_day)
    plt.grid()
    # pl.savefig("gradient_metaopt_5678676787656765456765.png")
    plt.show()

    fig = plt.figure()
    # ax = p3.Axes3D(fig)
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, np.zeros(np.shape(x)), zdir='z', s=100, c='r', antialiased=False)
    surf = ax.plot_surface(xx_new, yy_new, interp.T, rstride=5, cstride=5, cmap=cm.coolwarm,
                             linewidth=1)#, antialiased=False)
    #, depthshade=True)

    # cs = plt.contour(xx_new, yy_new, interp.T, rstride=1, cstride=1, color='g', cmap=cm.coolwarm)
    # pl.plot(maxes[::, 1], maxes[::, 2], "-ob", label='trajectory of maximum')
    # plt.clabel(cs, fmt='%.1f')  # , colors="black")
    # fig.colorbar(cs, shrink=0.5, aspect=5)
    plt.ylabel('white')
    plt.xlabel('red')
    # ax.set_zlabel('dCO2/dt *-1')
    plt.title(_day)
    # fig.legend(_day)
    plt.grid()
    # pl.savefig("gradient_metaopt_5678676787656765456765.png")
    plt.show()

    # import numpy as np
    # from mayavi import mlab
    # from matplotlib.cm import get_cmap  # for viridis

    # def f(x, y):
    #     return np.sin(2 * x) * np.cos(2 * y)
    #
    # # data for the surface
    # x = np.linspace(-2, 2, 100)
    # X, Y = np.meshgrid(x, x)
    # Z = f(X, Y)

    # data for the scatter
    # xx = 4 * np.random.rand(1000) - 2
    # yy = 4 * np.random.rand(1000) - 2
    # zz = f(xx, yy)

    fig = mlab.figure(bgcolor=(1, 1, 1))
    # note the transpose in surf due to different conventions compared to meshgrid
    # su = mlab.surf(xx_new, yy_new, f)
    su = mlab.surf(xx_new, yy_new, interp, colormap='RdYlBu', warp_scale="auto", representation='wireframe', line_width=0.5)
    mlab.outline(color=(0, 0, 0))
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5)
    axes.title_text_property.color = (0.0, 0.0, 0.0)
    axes.title_text_property.font_family = 'times'
    axes.label_text_property.color = (0.0, 0.0, 0.0)
    axes.label_text_property.font_family = 'times'
    # mlab.savefig("vector_plot_in_3d.pdf")
    mlab.gcf().scene.parallel_projection = True  # Source: <<https://stackoverflow.com/a/32531283/2729627>>.
    mlab.orientation_axes()
    # axes = mlab.axes(figure=su, x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True)
    # su = mlab.points3d(x, y, z)
    # sc = mlab.points3d(x, y, z_mark, scale_factor=0.1, scale_mode='none',
    #                     opacity=1.0, resolution=20, color=(1, 0, 0))

    # manually set viridis for the surface
    # cmap_name = 'viridis'
    # cdat = np.array(cm.get_cmap(cmap_name, 256).colors)
    # cdat = (cdat * 255).astype(int)
    # su.module_manager.scalar_lut_manager.lut.table = cdat

    mlab.show()


def get_one_day():
    pass



if __name__ == "__main__":
    db = {
        "host": '10.9.0.23',
        "user": 'remote_admin',
        "db": 'experiment',
        "password": "amstraLLa78x[$"
    }
    _day = datetime.datetime.now()
    search_table = [
        {"number": 1, "red": 130, "white": 130, "finished": 0, 'f': 0, 'q': 0},
        {"number": 2, "red": 70, "white": 190, "finished": 0, 'f': 0, 'q': 0},
        {"number": 3, "red": 190, "white": 70, "finished": 0, 'f': 0, 'q': 0},
        {"number": 4, "red": 40, "white": 160, "finished": 0, 'f': 0, 'q': 0},
        {"number": 5, "red": 160, "white": 40, "finished": 0, 'f': 0, 'q': 0},
        {"number": 6, "red": 100, "white": 100, "finished": 0, 'f': 0, 'q': 0},
        {"number": 7, "red": 220, "white": 220, "finished": 0, 'f': 0, 'q': 0},
        {"number": 8, "red": 25, "white": 235, "finished": 0, 'f': 0, 'q': 0},
        {"number": 9, "red": 145, "white": 115, "finished": 0, 'f': 0, 'q': 0},
        {"number": 10, "red": 85, "white": 55, "finished": 0, 'f': 0, 'q': 0},
        {"number": 11, "red": 205, "white": 175, "finished": 0, 'f': 0, 'q': 0},
        {"number": 12, "red": 55, "white": 85, "finished": 0, 'f': 0, 'q': 0},
        {"number": 13, "red": 175, "white": 205, "finished": 0, 'f': 0, 'q': 0},
        {"number": 14, "red": 115, "white": 145, "finished": 0, 'f': 0, 'q': 0},
        {"number": 15, "red": 235, "white": 25, "finished": 0, 'f': 0, 'q': 0},
        {"number": 16, "red": 17, "white": 138, "finished": 0, 'f': 0, 'q': 0}
    ]

    exp_id = 5

    show_optimal_points(db, search_table, exp_id, _day = '2020-11-25')