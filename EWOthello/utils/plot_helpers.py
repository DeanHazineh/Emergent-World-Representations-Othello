import matplotlib.pyplot as plt


def addAxis(thisfig, n1, n2, maxnumaxis=""):
    counterval = maxnumaxis if maxnumaxis else n1*n2
    axlist = [thisfig.add_subplot(n1, n2, i+1) for i in range(counterval)]
    return axlist
