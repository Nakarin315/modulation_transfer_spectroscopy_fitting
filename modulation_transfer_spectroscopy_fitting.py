import cv2
from PIL import Image
import numpy as np
from scipy import *
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special
from sklearn.metrics import r2_score

# Rescale picture
img = Image.open('modulation_transfer_spectroscopy_no.jpg')
img_scale = Image.open('modulation_transfer_spectroscopy_scale.jpg')
wsize = int(800) # x-axis pixel
hsize = int(500) # y-axis pixel
img = img.resize((wsize, hsize), Image.ANTIALIAS)
img_scale=img_scale.resize((wsize, hsize), Image.ANTIALIAS)
img.save('resized_modulation_transfer_spectroscopy_no.jpg')
img_scale.save('resized_modulation_transfer_spectroscopy_scale.jpg')
img = cv2.imread('resized_modulation_transfer_spectroscopy_no.jpg',2);
img_scale = cv2.imread('resized_modulation_transfer_spectroscopy_scale.jpg',2);

# Change image to black and white
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, bw_img_scale = cv2.threshold(img_scale,127,255,cv2.THRESH_BINARY)
bw_img = bw_img.transpose()


x_0= np.linspace(0,1, wsize);
y_b= np.linspace(1,1, wsize);
y_t= np.linspace(1,1, wsize);

for i in range(wsize):
    for j in range(hsize):
        if bw_img[i,j]==0:
            # Search data from bottom to top
            y_b[i]=hsize-j;  
        if bw_img[i,-j]==0:
            # Search data from top to bottom
            y_t[i]=j; 
y_0=(y_b+y_t)/2; # Average data 

# plt.plot(x_0,y_b,'--')
# plt.plot(x_0,y_0)
# plt.plot(x_0,y_t,'--')
###############################################################################
## How to recale the graph
## we can rescale the grahp by look at the minimum scale and maxmum scale
## In this case, minimum scale at index of 93 (int_min = 93, x=0) maximum scale at index of 758 (int_max = 758, x=10)
xmin = -1500;
xmax=1500;
int_xmin=70; # x= 0;
int_xmax=764; # x=10;

###############################################################################
## Plot data
###############################################################################
plt.figure(1)
x=np.linspace(xmin,xmax,int_xmax-int_xmin);
y=y_0[int_xmin:int_xmax]/hsize-1/2;
plt.plot(x,y)


###############################################################################
## Define function for fitting
###############################################################################

def Ln(delta,omega_m,n,Gamma):
    return Gamma**2/(Gamma**2+(delta-n*omega_m)**2);
def Dn(delta,omega_m,n,Gamma):
    return Gamma*(delta-n*omega_m)/(Gamma**2+(delta-n*omega_m)**2);
# This is the same function but set Bessel function as unity
def S_function(delta,s,s0,omega_m,Gamma,t):
    return (-s/np.sqrt(Gamma**2+omega_m**2))*((Ln(delta,omega_m,-1,Gamma)-Ln(delta,omega_m,-1/2,Gamma)+Ln(delta,omega_m,+1/2,Gamma)-Ln(delta,omega_m,1,Gamma))*np.cos(omega_m*t)+(Dn(delta,omega_m,1,Gamma)-Dn(delta,omega_m,1/2,Gamma)-Dn(delta,omega_m,-1/2,Gamma)+Dn(delta,omega_m,-1,Gamma))*np.sin(omega_m*t))+s0;

def S_fit(delta,s,s0):
    omega_m = 16.55;
    Gamma = 2*pi*32;
    t=0; # t=0 => in-phase signal, t =pi/(2*omega_m) => quadrature signal; 
    return S_function(delta,s,s0,omega_m,Gamma,t)

# Initial guesses for parameters
c0=[10,0.0]
# Fit curve with function 
c,cov = curve_fit(S_fit,x,y,c0)
# Define the fitting function
yp=S_fit(x,c[0],c[1])
print('Amplitude = %.2f '% (c[0]))
print('Offset =  %.2f '% (c[1]))
print('R^2 : %.5f'%(r2_score(y,yp)))

plt.figure(2)
plt.plot(x,y,alpha=0.5)
plt.plot(x,yp)