import matplotlib.pyplot as plt

def plotter(
        li: list,
        scatter: bool,
        show: bool):

    # plt.figure()
    cl =  ["green", "blue", "black"]
    for i in range(len(li)):
        x = li[i][1]
        y = li[i][0]

        if scatter:
            plt.scatter(x, y, marker="x", color=cl[i])
        else:
            plt.plot(x, y)

    if show:
        plt.show()

    return None
