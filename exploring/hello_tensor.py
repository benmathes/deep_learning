import theano
from theano import tensor

print( "declare two symbolic floating-point scalars")
a = tensor.dscalar()
b = tensor.dscalar()

print( "create a simple expression (+ is overloaded for tensor.dscalar)")
c = a + b


print( "convert the expression into a callable object that takes 2 values for a/b and computes c")
f = theano.function([a,b], c)


print( " bind a and b, then eval c.")
assert 3.5 == f(1.2, 2.3)
