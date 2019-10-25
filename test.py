import numpy as np

# Page 25

'''
X = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(X.shape)

random_state = 1
rgen = np.random.RandomState(random_state)
print(rgen)

w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
print(w_)

Y = np.array([0.6,-0.2,0.5,0.2,-0.5,-0.9])
tmp = np.where(Y >= 0.0, 1, -1)
print(tmp)

print(Y[0])
print(Y[1:])
print(Y[2:])
'''

# Page 32

x1_min = 0.0
x1_max = 3.0
x2_min = -2.0
x2_max = 1.0
res = 0.5
x1_arange = np.arange(x1_min,x1_max,res)
x2_arange = np.arange(x2_min,x2_max,res)
print("arange : ",x1_arange)
print("arange : ",x2_arange)
xx1, xx2 = np.meshgrid(x1_arange,x2_arange)
print("meshgrid : ",xx1)
print("meshgrid : ",xx2)
xx1_ra = xx1.ravel()
xx2_ra = xx2.ravel()
print("ravel : ",xx1_ra)
print("ravel : ",xx2_ra)
z1 = 
