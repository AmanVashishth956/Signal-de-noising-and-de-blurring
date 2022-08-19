# Team Member 1 : B20AI021 (Mitul Agrawal)
# Team Member 2 : B20MT005 (Aman Vashishth)

# After Running the Code, Close the Graph to Move to Next Graph
# Allow Computation Time after Closing Each Graph for Next Graph to Open (1-15 Seconds for Each)
# There are a total of 18 Graphs (Each Graph needs to be closed for the next Graph to be Displayed)
# Make sure data.csv is in same folder as the code. If it still shows error reading the data, enter full path in the line where the csv file is imported.

#============================================================================================================================================================

'''
Importing Required Libraries :
'''
    
# To Read CSV
import pandas as pd
# For Taking e^x and Getting Value of pi
import numpy as np
# To Plot Graphs for Visualizaton
import matplotlib.pyplot as plt

'''
Importing Temperature and Making List for Given Data :
'''

temp = pd.read_csv("data.csv",sep=",")

x = list(temp['x[n]'])
y = list(temp['y[n]'])

h = [1/16,4/16,6/16,4/16,1/16]

#============================================================================================================================================================

# Making Functions :

'''
Function : Standard Deviation
'''
def STD(x,y,brd=0) :
    x = x[brd:len(x)-1*brd] 
    y = y[brd:len(y)-1*brd]
    N = len(x) 
    if(N!=len(y)) : return -1
    s = 0
    for i in range (N) : 
        s = s + (x[i]-y[i])**2
    s = s / N
    s = np.sqrt(s)
    return s
# Square Root of Average of Square of Difference in Values of x and y     

'''
Function : Calculate Discrete Time Fourier Transformation for a Particular Angular Frequency
'''
def DTFT_W(x,w) : 
    Fw = 0 
    for n in range (len(x)) : 
        Fw = Fw + x[n]*np.exp(-1j*w*n)
    return Fw
# x is the input and w is the Angular Frequency for which we want to calculate the DTFT  

'''    
Function : Find a List of N values of Discrete Time Fourier Transformation with Angular Frequency in multiples of dw   
'''
def DTFT(x,N,dw) :
    F = []
    for n in range (N) : 
        F = F + [DTFT_W(x,n*dw)]
    return F
# dw is the step size we wish to take and N is the number of samples we want. 
# 2pi/dw samples are enough for our purpose since DTFT is periodic with period of 2pi
# for dw tending to 0 we will get a continuous DTFT
# We use the function defined above in this function to calculate DTFT for Angular Frequencies in multiples of dw            

'''
Function : Calculate Inverse Discrete Time Fourier Transformation for a Particular Time
'''
def IDTFT_N(F,n,dw) : 
    Xn = 0 
    N = int(2*np.pi/dw)
    for w in range (N) : 
        Xn = Xn + F[w]*np.exp(1j*n*(w*dw))*dw/(2*np.pi)
    return Xn
# F is the input and n is value for which we want to calculate the IDTFT
# for our purpose dw should be the same as the dw used for calculating DTFT      

'''
Function : Find a List of N values of Inverse Discrete Time Fourier Transformation 
'''
def IDTFT(F,N,dw) : 
    X = []
    for n in range (N) : 
        X = X + [IDTFT_N(F,n,dw).real]
    return X  
# dw is the step size we used for calculating DTFT
# We want N to be the same as the initial length of the input data list
# 2pi/dw samples are enough for our purpose since we want area within a period of 2pi        

'''
Function : Convolute Input with a Kernel with n0 being the index in the Kernel corresponding to n=0
'''
def Conv(x,h,n0=0) : 
    y = [0]*len(x)
    for i in range (len(x)) : 
        j1 = i
        if(j1>n0) : j1 = n0
        j1 = j1*(-1)
        j2 = len(x)-1-i
        if(j2>len(h)-1-n0) : j2 = len(h)-1-n0
        for j in range (j1,j2+1) : y[i] = y[i] + x[i+j] * h[n0+j]
    return y 
# x and h are the functions we want to convolute and n0 is the index in h corresponding to n=0 with respect to x
# This function will return convolution of any x and h in one dimension    

'''
Function : Denoise Data using Convolution Function Defined Above
'''
def Denoise(x,k=9) : 
    h = [1/k]*k 
    return Conv(x,h,k//2)
# h is the kernel we are using for denoising    
    
'''
Function : Deblur Data using the Functions Defined Above
'''
def Deblur(x,h,dw=0.001) : 

    n = len(x)
    N = int(2*np.pi/dw)+1 
    x = [0,0] + list(x) 
    Fx = DTFT(x,N,dw)
    Fh = DTFT(h,N,dw)
    F = [0]*N
    # Fx/Fh : 
    for i in range (N) : 
        xr = Fx[i].real
        xi = Fx[i].imag
        hr = Fh[i].real
        hi = Fh[i].imag
        hm = hr*hr + hi*hi
        cr = 0 
        cr = cr + (xr*hr + xi*hi)
        cr = cr + (1j)*(xi*hr - xr*hi)
        cr = cr/hm
        if(hm>0.25) :  F[i] = cr  
    f = IDTFT(F,n,dw)
    f = [i.real for i in f]
    return f 
# Lesser the dw, Lesser the error in the output, but computation time gets increased.
# Increase dw if Code is Taking a lot of time to compile   
# DTFT is Periodic with Period 2pi so we only need to find DTFT for Angular Frequency from 0 to 2pi 
# Since n=0 corresponds to index 2 in the Impulse Response, we add two zeroes at the start of x   
# We divide when |Fh[i]| > some value, otherwise Output will get Amplified and Unstable due to division by very small value.

#============================================================================================================================================================

'''
Denoising and Deblurring the Given Data : 
'''

# we only want to denoise and deblur data with full overlap with the kernel to deal with boundary effect
m = 4
y_start = y[:m]
y_end = y[-m:]

x1 = Deblur(Denoise(y),h) 
x2 = Denoise(Deblur(y,h)) 

x1 = y_start + x1[m:-m] + y_end
x2 = y_start + x2[m:-m] + y_end

'''
Analysing x1 and x2 : 
'''

std_y = STD(x,y)
std_x1 = STD(x,x1,7)
std_x2 = STD(x,x2,7)

print(f"Standard Deviation [x vs y]  : {std_y}")
print(f"Standard Deviation [x vs x1] : {std_x1}")
print(f"Standard Deviation [x vs x2] : {std_x2}")

#============================================================================================================================================================

'''
Plotting Data Points for Visualization : 
'''

# x1[n] and x2[n]

plt.scatter(range(len(x1)),x1,s=3)
plt.scatter(range(len(x2)),x2,s=3)
plt.legend(["x1[n]" , "x2[n]"])
plt.show()


# x1[n] and x2[n] continuous

plt.plot(range(len(x1)),x1)
plt.plot(range(len(x2)),x2)
plt.legend(["x1[n]" , "x2[n]"])
plt.show()


# x[n] and y[n] 

plt.scatter(range(len(x)),x,s=3)
plt.scatter(range(len(y)),y,s=3)
plt.legend(["x[n]" , "y[n]"])
plt.show()


# x[n] and x1[n] and x2[n]

plt.scatter(range(len(x)),x,s=3)
plt.scatter(range(len(x1)),x1,s=3)
plt.scatter(range(len(x2)),x2,s=3)
plt.legend(["x[n]" , "x1[n]" , "x2[n]"])
plt.show()


# x[n] and x1[n] and x2[n] continuous

plt.plot(range(len(x)),x)
plt.plot(range(len(x1)),x1)
plt.plot(range(len(x2)),x2)
plt.legend(["x[n]" , "x1[n]" , "x2[n]"])
plt.show()


# x[n] and y[n] and x1[n] and x2[n]

plt.plot(range(len(x)),x)
plt.plot(range(len(x1)),x1)
plt.plot(range(len(x2)),x2)
plt.plot(range(len(y)),y)
plt.legend(["x[n]" , "x1[n]" , "x2[n]" , "y[n]"])
plt.show()


# y[n] and Denoise(y[n])

y_denoise = Denoise(y)
plt.scatter(range(len(y)),y,s=3)
plt.scatter(range(len(y_denoise)),y_denoise,s=3)
plt.legend(["y[n]", "Denoise(y[n])"])
plt.show()


# y[n] anb Deblur(y[n])

y_deblur = Deblur(y,h)
plt.scatter(range(len(y)),y,s=3)
plt.scatter(range(len(y_deblur)),y_deblur,s=3)
plt.legend(["y[n]", "Deblur(y[n])"])
plt.show()


# dw vs Standard Deviation of [x[n] vs IDTFT(DTFT(x[n]))] 

w = [0.01*i for i in range(1,101)]
std_w = [STD(x,IDTFT(DTFT(x,int(10/i),i),len(x),i)) for i in w] 
plt.scatter(w,std_w,s=5)
plt.legend(["dw vs Standard Deviation of [x[n] vs IDTFT(DTFT(x[n]))]"])
plt.xlim(0,1.05)
plt.ylim(0,800)
plt.show()


# DTFT(x[n]) and DTFT(y[n]) in Argand Plane

Fx = DTFT(x,10000,0.01)
Fx_real = [i.real for i in Fx]
Fx_imag = [i.imag for i in Fx]
plt.scatter(Fx_real,Fx_imag,s=1)
Fy = DTFT(y,10000,0.01)
Fy_real = [i.real for i in Fy]
Fy_imag = [i.imag for i in Fy]
plt.scatter(Fy_real,Fy_imag,s=1)
plt.legend(["DTFT(x[n])","DTFT(y[n])"])
plt.show()


# DTFT(h[n]) in Argand Plane

Fh = DTFT(h,1000,0.01)
Fh_real = [i.real for i in Fh]
Fh_imag = [i.imag for i in Fh]
plt.scatter(Fh_real,Fh_imag,s=1)
plt.legend(["DTFT(h[n])"])
plt.show()


# DTFT(x[n]).real and DTFT(y[n]).real

Fx = DTFT(x,2000,0.01)
Fx_real = [i.real for i in Fx]
plt.scatter([i*0.01 for i in range(len(Fx_real))],Fx_real,s=1)
Fy = DTFT(y,2000,0.01)
Fy_real = [i.real for i in Fy]
plt.scatter([i*0.01 for i in range(len(Fy_real))],Fy_real,s=1)
plt.legend(["DTFT(x[n]).real","DTFT(y[n]).real"])
plt.xlim(0,20.1)
plt.show()


# DTFT(x[n]).imag and DTFT(y[n]).imag

Fx = DTFT(x,2000,0.01)
Fx_imag = [i.imag for i in Fx]
plt.scatter([i*0.01 for i in range(len(Fx_imag))],Fx_imag,s=1)
Fy = DTFT(y,2000,0.01)
Fy_imag = [i.imag for i in Fy]
plt.scatter([i*0.01 for i in range(len(Fy_imag))],Fy_imag,s=1)
plt.legend(["DTFT(x[n]).imag","DTFT(y[n]).imag"])
plt.xlim(0,20.1)
plt.show()


# Magnitude of DTFT(x[n]) and Magnitude of DTFT(y[n])

Fx = DTFT(x,2000,0.01)
Fx_mag = [np.sqrt(i.real**2+i.imag**2) for i in Fx]
plt.scatter([i*0.01 for i in range(len(Fx_mag))],Fx_mag,s=1)
Fy = DTFT(x,2000,0.01)
Fy_mag = [np.sqrt(i.real**2+i.imag**2) for i in Fy]
plt.scatter([i*0.01 for i in range(len(Fy_mag))],Fy_mag,s=1)
plt.legend(["Magnitude of DTFT(x[n])","Magnitude of DTFT(y[n])"])
plt.xlim(0,20.1)
plt.show()


# DTFT(h[n]).real

Fh = DTFT(h,2000,0.01)
Fh_real = [i.real for i in Fh]
plt.scatter([i*0.01 for i in range(len(Fh_real))],Fh_real,s=1)
plt.legend(["DTFT(h[n]).real"])
plt.xlim(0,20.1)
plt.show()


# DTFT(h[n]).imag

Fh = DTFT(h,2000,0.01)
Fh_imag = [i.imag for i in Fh]
plt.scatter([i*0.01 for i in range(len(Fh_imag))],Fh_imag,s=1)
plt.legend(["DTFT(h[n]).imag"])
plt.xlim(0,20.1)
plt.show()


# DTFT(h[n]).real and # DTFT(h[n]).imag

Fh = DTFT(h,2000,0.01)
Fh_real = [i.real for i in Fh]
Fh_imag = [i.imag for i in Fh]
plt.scatter([i*0.01 for i in range(len(Fh_real))],Fh_real,s=1)
plt.scatter([i*0.01 for i in range(len(Fh_imag))],Fh_imag,s=1)
plt.legend(["DTFT(h[n]).real" , "DTFT(h[n]).imag"])
plt.xlim(0,20.1)
plt.show()


# Magnitude of DTFT(h[n])

Fh = DTFT(h,2000,0.01)
Fh_mag = [np.sqrt(i.real**2+i.imag**2) for i in Fh]
plt.scatter([i*0.01 for i in range(len(Fh_mag))],Fh_mag,s=1)
plt.legend(["Magnitude of DTFT(h[n])"])
plt.xlim(0,20.1)
plt.ylim(0,1.05)
plt.show()


#============================================================================================================================================================

'''
Output : 

Standard Deviation [x vs y]  : 1.435669673478948
Standard Deviation [x vs x1] : 1.001968956085593
Standard Deviation [x vs x2] : 0.9507238799155275
'''