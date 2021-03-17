import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)
N = 6
x = np.array([1,2,3,4,2,1])
h = (-0.5)**np.arange(N)
for i in range(N):
	if i > 1:
		h[i] += np.power(-0.5,i-2)
		
		
def dft_matrix(n):
	dft_mat = np.zeros((n,n),dtype=np.complex128)
	for i in range(n):
		for j in range(n):
				dft_mat[i][j] = np.exp(-2j*np.pi*i*j/n)
	return dft_mat
	
	
def dft(x):
	n = len(x)
	F = dft_matrix(n)
	return F
	
	
X = dft(x)
H = dft(h)
Y = X*H
plt.figure(figsize=(9,15))
plt.subplot(3,2,1)
plt.stem(np.abs(X),use_line_collection=True)
plt.title('$|X(k)|$')
plt.grid()

plt.subplot(3,2,2)
plt.stem(np.angle(X),use_line_collection=True)
plt.title(r'$\angle{X(k)}$')
plt.grid()

plt.subplot(3,2,3)
plt.stem(np.abs(H),use_line_collection=True)
plt.title('$|H(k)|$')
plt.grid()

plt.subplot(3,2,4)
plt.stem(np.angle(H),use_line_collection=True)
plt.title(r'$\angle{H(k)}$')
plt.grid()

plt.subplot(3,2,5)
plt.stem(np.abs(Y),use_line_collection=True)
plt.title('$|Y(k)|$')
plt.grid()

plt.subplot(3,2,6)
plt.stem(np.angle(Y),use_line_collection=True)
plt.title(r'$\angle{Y(k)}$')
plt.grid()
plt.savefig('../figs/EE18BTECH11016.pdf')
plt.savefig('../figs/EE18BTECH11016.eps')

plt.subplots_adjust(hspace=0.5)
plt.show()
