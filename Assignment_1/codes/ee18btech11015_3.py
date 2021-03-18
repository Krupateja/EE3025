import time
import numpy as np
import matplotlib.pyplot as plt

def DFT(x):
	n = len(x)
	F = np.zeros((n,n),dtype=np.complex128)
	for i in range(n):
		for j in range(n):
				F[i][j] = np.exp(-2j*np.pi*i*j/n)
	return F@x

def FFT(x):	
	n = len(x)
	
	if(n == 1):
		return x
	
	elif(n == 2):
		return np.hstack((x[0]+x[1],x[0]-x[1]))
	
	X1 = FFT(x[::2])
	X2 = FFT(x[1::2])
	
	D = np.zeros((n//2,), dtype=np.complex128)
	for i in range(n//2):
		D[i] = np.exp(-2j*np.pi*i/n)
	
	X_u = X1 + D*X2
	X_l = X1 - D*X2
	
	return np.hstack((X_u,X_l))

DFT_time = np.zeros(12)
FFT_time = np.zeros(12)

for i in range(12):
	N = 2**i
	x = np.random.randint(1,5,size=N)
	t1 = time.time()
	X_d = DFT(x)
	t2 = time.time()
	X = FFT(x)
	t3 = time.time()
	DFT_time[i] = t2-t1
	FFT_time[i] = t3-t2
	
axis = 2**np.arange(12)
plt.plot(axis, DFT_time, label = 'DFT Computation Time')
plt.plot(axis, FFT_time, label = 'FFT Computation Time')
plt.title('Computation time comparison of DFT and FFT')
plt.xlabel('N')
plt.ylabel('Execution time (in s)')
plt.xscale('log', basex=2)
plt.grid()
plt.legend()
plt.show()
