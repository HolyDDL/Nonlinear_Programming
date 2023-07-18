import numpy as np

class Particle():
    
    def __init__(self, x0, v0, vmax:np.array, objfun, w=0.9, c1=2, c2=2, max_iter_times=50) -> None:
        '''
            ## Parameters:
                x0: the first iter point

                v0: the first direction step

                vmax: the maximum direction step

                objfun: the object function

                w: inertia weight

                c1: acceleration coefficient 1

                c2: acceleration coefficient 2
                
                max_iter_times: the maximum iter times
        '''
        self.dim = len(x0)
        self.w = w
        self.vmax = np.array(vmax)
        self.c1 = c1
        self.c2 = c2
        self.x = np.array(x0)
        self.v = np.array(v0)
        self.objfun = objfun
        self.pBest_value = self.calculate_objvalue(self.objfun, *self.x)
        self.pBest = np.array(x0)
        self.gBest = self.pBest
        self.max_iter_times = max_iter_times

    def calculate_objvalue(self, objfun, *array) ->float:
        return objfun(array)
    
    def iter(self, xk):
        r1 = np.random.rand()
        r2 = np.random.rand()
        next_v = self.w * self.v + self.c1*r1*(self.pBest - xk) + self.c2*r2*(self.gBest - xk)
        for i in range(len(next_v)):
            if next_v[i] > self.vmax[i]:
                next_v[i] = self.vmax[i]
        next_x = xk + next_v
        self.x = next_x
        self.v = next_v
        temp = self.calculate_objvalue(self.objfun, *self.x)
        if temp < self.pBest_value:
            self.pBest_value = temp
            self.pBest = self.x

def PSO(numbers, objfun, x0s, v0s, vmax, w=0.9, c1=2, c2=2, max_tier_times=3000):
    '''
        ## Parameters:
            numbers: the number of partcles

            objfun: the object function

            x0s: the first iter point matrix, size: [numbers, dim]

            v0s: the first direction step maxtirx, size: [numbers, dim]

            vmax: the maximum direction step

            w: inertia weight

            c1: acceleration coefficient 1

            c2: acceleration coefficient 2

            max_iter_times: the maximum iter times
        
        ## Returns:
            global_gBest: the best solution

            global_gBest_value: the value of object function in the best solution

            iterpoints: all of the iterated points
    '''
    particles = []
    for i in range(numbers):
        particles.append(Particle(x0s[i], v0s[i], vmax, objfun, w, c1, c2,max_tier_times))
    global_gBest_value = particles[0].pBest_value
    global_gBest = particles[0].pBest
    times = 0
    iterpoints = np.zeros(particles[0].max_iter_times)
    iter_w_scale = (w-0.4) / max_tier_times
    while(1):
        # iter the inertia weight
        w = w - iter_w_scale
        if times == 0:
            gBest_value = particles[0].pBest_value
            gBest = particles[0].pBest
            for p in particles:
                p.gBest = gBest
        for particle in particles:
            particle.iter(particle.x)
            if particle.pBest_value < gBest_value:
                gBest_value = particle.pBest_value
                gBest = particle.pBest
                for p in particles:
                    p.gBest = gBest
            
        # if(np.abs(global_gBest_value - gBest_value) <= gap):
        #     global_gBest = gBest
        #     break
        # else:
        global_gBest_value = gBest_value
        global_gBest = gBest
        if times >= particles[0].max_iter_times:
            break
        iterpoints[times] = global_gBest_value
        times = times + 1
    print(f'times = {times}')
    return global_gBest, global_gBest_value, iterpoints

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def objfun(xs):
        f = 1
        for i,x in enumerate(xs):
            f = f*np.cos(x / np.sqrt(i+1))
        for x in xs:
            f = f + x / 4000
        return f+1
    x0s = np.random.rand(10,5)
    v0s = np.random.rand(10,5)
    vmax = 10*np.ones(5)
    [point, value, points] = PSO(3, objfun, x0s, v0s, vmax)
    times = range(len(points))
    plt.plot(times, points)
    print(point)
    print(value)
    plt.show()
