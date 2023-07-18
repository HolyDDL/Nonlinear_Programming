# -----------------------------------------
# @Decription: This is the python file of some common functions that may be used in other files
# @Author: Yiwei Ren.
# @Date: July 09, 2023, Sunday 22:33:34
# @Copyright (c) 2023 Yiwei Ren.. All rights reserved.
# -----------------------------------------


def firstDiff(fun, x0, e=1e-6):
    import numpy as np
    x0 = np.array(x0)
    return (fun(x0+e) - fun(x0-e)) / (2*e)

def secondDiff(fun, x0, e=1e-3):
    import numpy as np
    x0 = np.array(x0)
    return (fun(x0+e) + fun(x0-e) - 2*fun(x0)) / (e*e)

def modulus(x):
    import numpy as np
    x = np.array(x)
    return np.sqrt(x @ x)

def gradient(fun, x0, e=1e-6):
    import numpy as np
    grad = np.zeros(len(x0))
    for dim in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[dim] += e
        xm[dim] -= e
        grad[dim] = (fun(xp) - fun(xm)) / (2*e)
    return grad
