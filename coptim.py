# -----------------------------------------
# @Decription: This is the python file to solve constrained optimization problems
# @Author: Yiwei Ren.
# @Date: July 12, 2023, Wednesday 20:00:07
# @Copyright (c) 2023 Yiwei Ren.. All rights reserved.
# -----------------------------------------

import numpy as np


class ExtenalPenaltyFunction():

    def __init__(self, ori_objfun, inequals:list, equals:list, c=0.1, alpha=2) -> None:
        '''
            ## Parameters:

                ori_objfun: the object function that should be minimized
                inequals: the inequality constraints list
                equals: the equality constrains list
                c: the penalty weight. Default=0.1
                alpha: the increase factor of penalty weight. Default=2
            
            ## Description:

                The class using extenal penalty function method to solve constrained optimization. 
                It have PSO method, Conjugate Gradient method, Deepest Descent method to solve the problem.
        '''
        self.ori_objfun = ori_objfun
        self.inequals = inequals
        self.equals = equals
        self.c = c
        self.alpha = alpha
        self.unlock = True

    def epf(self, x):
        value = self.ori_objfun(x)
        for inequal in self.inequals:
            temp = inequal(x)
            if temp > 0:
                value += self.c * temp**2
        for equal in self.equals:
            value += 0.5 * self.c * equal(x)**2
        if self.unlock:
            self.iter_c()
        return value
    
    def S(self, x):
        value = 0
        for inequal in self.inequals:
            temp = inequal(x)
            if temp > 0:
                value += temp**2
        for equal in self.equals:
            value += equal(x)**2
        return value
    
    def iter_c(self):
        self.c = self.alpha * self.c

    def PSO_method(self, numbers, x0s, v0s, vmax ,w= 0.9, c1= 2, c2= 2, max_tier_times= 3000, neediterpoints=None):
        '''
            ## Parameters:

                numbers: the number of partcles
                x0s: the first iter point matrix, size: [numbers, dim]
                v0s: the first direction step maxtirx, size: [numbers, dim]
                vmax: the maximum direction step, size: [dim]
                w: inertia weight
                c1: acceleration coefficient 1
                c2: acceleration coefficient 2
                max_iter_times: the maximum iter times
                neediterpoints: whether need return all iter points or not. Type anything to enable it. Default=None

            ## Returns:

                solution: the solution of the problem
                iterpoints: all iterated points if neediterpoints is enabled

            ## Descriptions:

                Use Particle Swarm Optimization algorithm to solve the problem
        '''
        import pso
        solution, _, iterpoints = pso.PSO(numbers, self.epf, x0s, v0s, vmax, w, c1, c2, max_tier_times)
        if neediterpoints:
            return solution, iterpoints
        else:
            return solution
    
    def ConjugateGradient_method(self, x0, eps=1e-6, e = 1e-6, method = 'PRP', tmin = 0, tmax = 1, onedime = 1e-6, maxtimes=3000):
        '''
            ## Parameters:
                
                x0: the initial point
                eps: the iteration precision of whole algorithm. Default=1e-6
                e: iteration precision of CG. Default=1e-6
                method: the method of calculating lambda. Default='PRP'. Options='FR'
                tmin: the lower bound of the golden section one-dimensional search interval. Default=0
                tmax: the upper bound of the golden section one-dimensional search interval. Default=1
                onedime: one-dimensional search precison. Default=1e-6
                maxtimes: the maximum iterable times
            
            ## Returns:

                x: the solution of the problem
            
            ## Description: 

                Use Conjugate Gardient method to solve the problem
        '''
        from ucoptim import ConjugateGradient
        self.unlock = False
        times  = 0
        x = x0
        while(1):
            x = ConjugateGradient(self.epf, x, e, method, tmin, tmax, onedime)
            if not x:
                return None
            if self.S(x) <= eps:
                return x
            self.iter_c()
            times += 1
            if times >= maxtimes:
                return x

    def SteepestDescent_method(self, x0, eps=1e-6, e = 1e-6, tmin = 0, tmax = 1, onedime = 1e-6, maxtimes=3000):
        '''
            ## Parameters:

                x0: the initial point
                eps: the iteration precision of whole algorithm. Default=1e-6
                e: iteration precision. Default=1e-6
                tmin: the lower bound of the golden section one-dimensional search interval. Default=0
                tmax: the upper bound of the golden section one-dimensional search interval. Default=1
                onedime: one-dimensional search precison. Default=1e-6
                maxtimes: the maximum iterable times

            ## Returns:

                x: the solution of the problem
            
            ## Description: 

                Use Steepest Descent method to solve the problem
        '''
        from ucoptim import SteepestDescent
        self.unlock = False
        times = 0
        x = x0
        while(1):
            x = SteepestDescent(self.epf, x0, e, tmin, tmax, onedime)
            if not x:
                return None
            if self.S(x) <= eps:
                return x
            self.iter_c()
            times += 1
            if times >= maxtimes:
                return x


if __name__ == '__main__':
    def ori_objfun(x):
        x = np.array(x)
        value = 0.0
        for each in x:
            value += x**2
        return value
    def inequal(x):
        x = np.array(x)
        value = 0.0
        for each in x:
            value = 1-x
        return value
    pf = ExtenalPenaltyFunction(ori_objfun, [inequal],[])
    # solution, minvalue = pf.PSO_method(3, [[5], [6], [7]], np.random.rand(3,1), [1,1,1])
    # print(solution)
    # print(pf.SteepestDescent_method([3]))
