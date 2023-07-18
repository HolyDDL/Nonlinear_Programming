# -----------------------------------------
# @Decription: This is the python file about one-dimensional search.
# @Author: Yiwei Ren.
# @Date: July 08, 2023, Saturday 22:22:20
# @Copyright (c) 2023 Yiwei Ren.. All rights reserved.
# -----------------------------------------

def GoldenSection(fun, tmin, tmax, e=1e-6) -> float:
    '''
        ## Parameters:
            fun: the object function that shuould to be minimized
            tmin: the lower bound of the search interval
            tmax: the upper bound of the search interval
            e: interval precision. Default=1e-6
        ## Returns:
            t: the soluction of one dim search
            None: no solution
        ## Description:
            Use gorden section (0.618) method to solve one-dimensional search problems.
    '''
    t1 = tmax - 0.618*(tmax - tmin)
    t2 = tmin + 0.618*(tmax - tmin)
    ft1 = fun(t1)
    ft2 = fun(t2)
    while(1):
        if ft1 <= ft2:
            if t2 - tmin <= e:
                return t1
            else:
                tmax = t2
                t2 = t1
                t1 = tmax - 0.618*(tmax - tmin)
                ft2 = ft1
                ft1 = fun(t1)
        else:
            if tmax - t1 <= e:
                return t2
            else:
                tmin = t1
                t1 = t2
                t2 = tmin + 0.618*(tmax - tmin)
                ft1 = ft2
                ft2 = fun(t2)
    return None


def NewtonIter(fun, t0, e=1e-6):
    '''
        ## Parameters:
            fun: the object function that shuould to be minimized
            x0: the first iter point
            e: iter presicion. Default=1e-6
        ## Returns:
            t: the soluction of one dim search
            None: no solution
        ## Decription:
            Use Newton's iteration method to solve one-dimensional problems
    '''
    from utils import firstDiff, secondDiff
    t = t0
    while(1):
        fd = firstDiff(fun, t)
        if abs(fd) < e:
            return t
        else:
            sd = secondDiff(fun, t)
            if abs(sd - 0.0) < 1e-12:
                return None
            else:
                t -= fd/sd
                if abs(fd/sd) < e:
                    return t
    return None
                
if __name__ == '__main__':
    import numpy as np
    def fun(t):
        return np.arctan(t)*t - 0.5*np.log(t*t + 1)
    print(NewtonIter(fun,1))
    print(GoldenSection(fun, -1,1))