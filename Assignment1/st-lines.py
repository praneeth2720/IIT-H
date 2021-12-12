import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

A = np.array([-2,-2])
B = np.array([2,-4])
#the line joining A and B divided by the point P in ratio of 3:4

#according to the section formula 
P = (4*A + 3*B) / 7
print(P) 

# therefore the point P is (-0.285, -2.857)
P = np.array([-0.285, -2.857])

def dir_vec(I,J):
  return I-J

def norm_vec(I,J):
  return omat*dir_vec(I,J)


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

x_AB = line_gen(A,B)



#Triangle
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')


#Orthocenters
plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.axis('equal')
plt.grid()
plt.show()
