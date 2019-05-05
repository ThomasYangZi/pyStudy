from svmutil import *
from svm import *

# y, x = [1, -1], [{1: 1, 2: 1}, {1: -1, 2: -1}]
y, x = svm_read_problem("2001.csv")
print(y, x)
prob = svm_problem(y, x, True)
param = svm_parameter('-t 2 -c 70 -b 0 -g 0.1')
model = svm_train(prob, param)
yt = [1]
xt = [{1: 1, 2: 1}]
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(p_label)