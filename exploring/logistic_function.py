import theano
import theano.tensor as tensor
from pprint import pprint

matrix = tensor.dmatrix('matrix')

logistic_expression = 1 / (1 + tensor.exp(-matrix))
logistic_function = theano.function([matrix], logistic_expression)

pprint(
    logistic_function([[0, 1],
                       [-1, -2]]))



a, b = tensor.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])
pprint(f([[1, 1], [1, 1]], [[0, 1], [2, 3]]))


