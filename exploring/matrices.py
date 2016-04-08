from pprint import pprint

import theano
from theano import tensor

def add_matrices():
    x = tensor.dmatrix('x')
    y = tensor.dmatrix('y')
    z = x + y
    f = theano.function([x, y], z)
    
    pprint(f(
        [[1, 2],
         [3, 4]],
        [[10, 20],
         [30, 40]]
    ))


def mult_matrices():
    a = tensor.vector()
    b = tensor.vector()
    binomial_expansion = a**2 + b**2 + 2*a*b
    f = theano.function([a, b], binomial_expansion)
    pprint(f([0,1,2], [3,4,5]))


mult_matrices()    
