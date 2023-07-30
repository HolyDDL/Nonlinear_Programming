# Nonlinear Programming

**This project included three parts:**

1. One-dimensional search methodone(see as [dimsearch.py](onedimsearch.py))
2. Method of unconstrained optimization problems(see as [ucoptim.py](ucoptim.py))
3. Method of constrained optimization problems(see as [coptim.py](coptim.py))

---

## Part 1: One-dimensional Search

- included 2 methods: GoldenSection Method(0.618 Method), NewtonIteration Method

- **How to use it:**
  
  1. Define a object function that should be minimized
  2. Call method function to solve it

- ***DEMO***
  
  ```py
    from onedimsearch import *
    # define a object function
    def fun(t):
        return np.arctan(t)*t - 0.5*np.log(t*t + 1)
    # call method function
    print(NewtonIter(fun,1))
    print(GoldenSection(fun, -1,1))
    ```

---

## Part 2: Unconstrained optimization problems

- include 2 methods: SteepestDescent Method, ConjugateGradient Method
- **How to use it:**
  
  1. Define a object function that should be minimized
  2. Call method function to solve it
- ***DEMO***

  ```py
    from ucoptim import *
    # define a object function
    def fun(x):
        return x[0]**2 + 25*x[1]**2
    # call method function
    print(ConjugateGradient(fun, [32,2],method='FR'))
    print(SteepestDescent(fun, [1,2]))
  ```

---

## part 3: Constrained optimization problems

- include 2 method: ExtenalPenaltyFunction Method(including 3 methods to solve), InternalPenaltyFunction Method(Not be ready yet)
- **How to use it:**
  
  1. Define a object function that should be minimized and constraints
  2. Instantiate Extenal/Internal class
  3. Use class methods to solve it

- ***DEMO***
  
  ```py
    from coptim import *
    # define object function
      def ori_objfun(x):
        x = np.array(x)
        value = 0.0
        for each in x:
            value += each**2
        return value
    # define inequal constraint function
    def inequal(x):
        x = np.array(x)
        value = 0.0
        for each in x:
            value = 1-each
        return value
    # use extenal class to solve
    pf = ExtenalPenaltyFunction(ori_objfun, [inequal],[])
    # use extenal.PSO method to solve
    solution, minvalue = pf.PSO_method(3, [[5], [6], [7]], np.random.rand(3,1), [1,1,1])
    print(solution)
    # use extenal.SteepestDescent method to solve
    # print(pf.SteepestDescent_method([3]))
    ```
