import matplotlib.pyplot as plt

def plotter(
        li: list,
        scatter: bool,
        show: bool):

    # plt.figure()
    for i in range(len(li)):
        x = li[i][1]
        y = li[i][0]

        if scatter:
            plt.scatter(x, y, marker="x")
        else:
            plt.plot(x, y)

    if show:
        plt.show()

    return None
