import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Eq, solve, Matrix, Symbol
import matplotlib.pyplot as plt

k = symbols('k')
#given three points on the coordinate
#P = np.array([k,3])
#Q = np.array([6,-2])
#R = np.array([-3,4])
P = np.array([k,3,1])
Q = np.array([6,-2,1])
R = np.array([-3,4,1])
X = np.array([P,Q,R])
print(X)
#if the points are collinear then the area of the triangle must be zero 
A = Matrix(X)
print(A.det())

# defining equation
eq1 = Eq(((-6)*k), 9)
print("Equation 1:")
print(eq1)
print(solve((eq1), (k)))


P = np.array([-3/2 ,3])
Q = np.array([6,-2])
R = np.array([-3,4])

def line_gen(I,J):
  len =10
  x_IJ = np.zeros((2,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = I + lam_1[i]*(J-I)
    x_IJ[:,i]= temp1.T
  return x_IJ

I = np.array([-2,-2]) 
J = np.array([1,3]) 
dvec = np.array([-1,1]) 
omat = np.array([[0,1],[-1,0]])

x_RQ = line_gen(R,Q)



#Triangle
plt.plot(x_RQ[0,:],x_RQ[1,:],label='$RQ$')


plt.plot(R[0], R[1], 'o')
plt.text(R[0] * (1 + 0.1), R[1] * (1 - 0.1) , 'R')
plt.plot(Q[0], Q[1], 'o')
plt.text(Q[0] * (1 - 0.2), Q[1] * (1) , 'Q')

plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.axis('equal')
plt.grid()
plt.show()

