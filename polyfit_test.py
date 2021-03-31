import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates

if __name__ == "__main__":
    # x = np.array([0, 1, 2, 3, 4, 5])
    xdates = [
        dt.datetime(2021, 3, 15, 0, 0, 0),
        dt.datetime(2021, 3, 16, 0, 0, 0),
        dt.datetime(2021, 3, 17, 0, 0, 0),
        dt.datetime(2021, 3, 18, 0, 0, 0),
        dt.datetime(2021, 3, 19, 0, 0, 0),
        dt.datetime(2021, 3, 20, 0, 0, 0),
    ]

    x = mdates.date2num(xdates)

    y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    y_err = np.array([0.05, 0.05, 0.05, 0.05, 0.5, 0.05])
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)

    xp = np.linspace(np.min(x), np.max(x), 100)
    dxp = mdates.num2date(xp)
    print(z)
    fig = plt.figure()
    # plt.plot(xdates, y, "og", label='raw data')
    plt.plot(dxp, p(xp), "-b", label='approx data')
    plt.errorbar(xdates, y, y_err, marker='.',
             color='k',
             ecolor='k',
             markerfacecolor='b',
             label="series 1",
             capsize=0,
             linestyle='')

    plt.grid()
    plt.show()

    # example data
    x = np.arange(0.1, 4, 0.5)
    y = np.exp(-x)

    # example error bar values that vary with x-position
    error = 0.1 + 0.2 * x

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.errorbar(x, y, yerr=error, fmt='-o')
    ax0.set_title('variable, symmetric error')

    # error bar values w/ different -/+ errors that
    # also vary with the x-position
    lower_error = 0.4 * error
    upper_error = error
    asymmetric_error = [lower_error, upper_error]

    ax1.errorbar(x, y, xerr=asymmetric_error, fmt='o')
    ax1.set_title('variable, asymmetric error')
    ax1.set_yscale('log')
    plt.show()



