# -----------------------------------------
# @Decription: This is the python file to solve unconstrained optimization problems
# @Author: Yiwei Ren.
# @Date: July 09, 2023, Sunday 19:36:46
# @Copyright (c) 2023 Yiwei Ren.. All rights reserved.
# -----------------------------------------


def SteepestDescent(objfun, x0, e=1e-6, tmin=0,tmax=1, onedime=1e-6):
    '''
        ## Parameters:
            objfun: the object function that should be minimized
            x0: the initial point
            e: iteration precision. Default=1e-6
            tmin: the lower bound of the golden section one-dimensional search interval. Default=0
            tmax: the upper bound of the golden section one-dimensional search interval. Default=1
            onedime: one-dimensional search precison. Default=1e-6
        ## Returns:
            x: the solution of unconstrained optimization problem
            None: no solution
        ## Description:
            Use steepest descent method and golden section one-dimensional search method to solve unconstrained optimization problems.
    '''
    from onedimsearch import GoldenSection
    from utils import gradient, modulus
    x = x0
    while(1):
        grad = gradient(objfun, x)
        if modulus(grad) < e:
            return x
        else:
            p = -grad
            iterfun = lambda t: objfun(x+t*p)
            t = GoldenSection(iterfun, tmin, tmax, onedime)
            x += t*p
    return None

def ConjugateGradient(objfun, x0, e=1e-6, method='PRP', tmin=0, tmax=1, onedime=1e-6):
    '''
        ## Parameters:
            objfun: the object function that should be minimized
            x0: the initial point
            e: iteration precision. Default=1e-6
            method: the method of calculating lambda. Default='PRP'. Options='FR'
            tmin: the lower bound of the golden section one-dimensional search interval. Default=0
            tmax: the upper bound of the golden section one-dimensional search interval. Default=1
            onedime: one-dimensional search precison. Default=1e-6
        ## Returns:
            x: the solution of unconstrained optimization problem
            None: no solution
        ## Description:
            Use conjugate gradient method and golden section one-dimensional search method to solve unconstrained optimization problems.
            Can choose PRP or FR method to calculate lambda.
            In some problems, one of method of calculating lambda may not converge.
    '''
    from onedimsearch import GoldenSection
    from utils import modulus, gradient
    if method == 'FR':
        def lam(gk, gkp):
            return modulus(gkp)**2 / modulus(gk)**2
    else:
        def lam(gk, gkp):
            return gkp.dot(gkp - gk) / modulus(gk)
    x = x0
    while(1):
        k = 0
        gradk = gradient(objfun, x)
        if modulus(gradk) < e:
            return x
        p = -gradk
        while(1):
            iterfun = lambda t: objfun(x+t*p)
            t = GoldenSection(iterfun, tmin, tmax, onedime)
            x += t*p
            gradkp = gradient(objfun, x)
            if modulus(gradkp) < e:
                return x
            else:
                if k+1 == len(x0):
                    break
                else:
                    p = -gradient(objfun, x) + lam(gradk, gradkp) * p
                    k += 1
            gradk = gradient(objfun, x)
    return None

        
if __name__ =='__main__':
    def fun(x):
        return x[0]**2 + 25*x[1]**2
    print(ConjugateGradient(fun, [32,2],method='FR'))
    print(SteepestDescent(fun, [1,2]))