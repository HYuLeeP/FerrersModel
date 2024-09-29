
import matplotlib.pyplot as plt
# import matplotlib
import scipy.special as sp
from scipy.integrate import odeint
from scipy.optimize import brentq
import config
import numpy as np

""" dphi(z,t) calculates the derivatives for the equations of motion in the (x,y) plane
for a rotating Ferrers bar. Returns vx, vy, force_x, force_y.  The forces include the
Coriolis and centrifugal components.

It needs some global 
parameters for the bar and disk which 
I put in at the start of the main program  e.g.

# ------------------------------
e = 0.2                                         #  bar minor axis c = e*(bar major axis a)
a = 1/np.sqrt(1 - e**2)                         #  bar semilength a   a^2 - c^2 = 1
c = e*a
eps = 5.                                        # eps is a.sqrt(1-e^2) = physical scale for bar in kpc

# set parameters for potential:   V is softened flat rotation curve, 

AA = 0.5                                        # disk potential is 0.5 V^2 log(r^2 + AA^2)
Mbar = 2.0e+10                                  # bar mass units M_sun
V = 240.                                        # units km/s
P = 2.                                          
Q = P*0.98*(Mbar/2e+10)*(5./eps)*((240/V)**2)

# with these parameters, L4 = CR is at y = 1.4737 

all lengths in units of eps
times in units of 1/omega_bar
velocities in units of omega_bar.eps

# ------------------------------  """

def dphi(z,t):

    e = config.e
    a = 1/np.sqrt(1 - e**2)
    c = e*a
    AA = config.AA      # disk potential is 0.5 V^2 log(r^2 + AA^2)
    Mbar = config.Mbar  # units M_sun
    V = config.V        # units km/s
    eps = config.eps    # units kpc
    P = config.P        # P = 1,2 => CR at r = eps, 2eps
    Q = P*0.98*(Mbar/2e+10)*(5./eps)*((240/V)**2)


    z0, z1, z2, z3 = z
    x = z0
    y = z1
    
    # -----------------------------------

    def fb(v):                  # fn for solving for v via brentq
        g = (x**2)*(v**2)*(1-(v**2)) + ((y**2)+1)*(v**2)-1
        return g
    
    # -----------------------------------
    
    if ((x/a)**2 + (y/c)**2 <= 1.):   # x,y within ellipsoid figure
        v = 1/a    
#       print 'line 67 v = ',v                   
    else:                             # x,y outside ellipsoid figure
        u = np.linspace(0,1,11)
        g = (x**2)*(u**2)*(1-(u**2)) + ((y**2)+1)*(u**2) -1 
        q = [i for i in range(0,10) if (g[i]*g[i+1] <= 0)][0] 
        aa = u[q]
        bb = u[q+1]
        v = brentq(fb,aa,bb)  
#       print 'line 75 v = ',v
        
    I0 = 2*v
    I1 = (v**3)/1.5
    I2 = (v**5)/2.5
    J = np.log((1.+v)/(1.-v))
    J0 = (2.*v/(1.-v**2.) + J)/2.
    J1 = (2.*(v+v**3)/(1.-v**2)**2 - J)/8.

    W10 = J
    W20 = J0 - W10
    W11 = W10 - I0
    W30 = J1 - W20
    W12 = W11 - I1
    W21 = W20 - W11
    W40 = (66.*v**5 - 80.*v**3 -15.* ((1-v**2)**3)*J + 30.*v)/(48.*((1-v**2)**3))  
    W22 = W21 - W12
    W31 = W30 - W21
    W13 = W12 - I2
    
    # calculate forces
    
    if ((x/a)**2 + (y/c)**2 <= 1.):  # inside figure
       s1 = -2.*x*W11
       s2 = 4.*(y**2)*x*W21 + 4.*(x**3)*W12
       s3 = -(2.*(y**4)*x*W31 + 4.*(y**2)*(x**3)*W22 + 2.*(x**5)*W13)
       dPhidx = s1 + s2 + s3
       s1 = -2.*y*W20
       s2 = 4.*(y**3)*W30 + 4.*y*(x**2)*W21    
       s3 = -(2.*(y**5)*W40 + 4.*(y**3)*(x**2)*W31 + 2.*y*(x**4)*W22)   
       dPhidy = s1 + s2 + s3
       
    else:                            # outside figure
        denom = (x**2)*(1-2.*(v**2)) + y**2 +1
        dvdx = -x*v*(1-v**2)/denom
        dvdy = -y*v/denom
    
        dw10dv = 2./(1-v**2)
        dw10dx = dw10dv*dvdx
        dw10dy = dw10dv*dvdy
    
        dw20dv = 2.*(v**2)/((1-v**2)**2)
        dw11dv = 2.*(v**2)/(1-v**2)
        dw30dv = 2.*(v**4)/((1-v**2)**3)
        dw12dv = dw11dv - 2.*v**2
        dw21dv = dw20dv - dw11dv
        dw40dv = 2.*(v**6)/((1-v**2)**4)
        dw22dv = dw21dv - dw12dv    
        dw31dv = dw30dv - dw21dv
        dw13dv = dw12dv - 2.*v**4
    
    # calculate dPhidx
    
        t0 = dw10dv/3.          # t-terms involve derivatives of Wij
        t1 = -((y**2)*dw20dv + (x**2)*dw11dv)
        t2 = (y**4)*dw30dv + 2.*(y**2)*(x**2)*dw21dv + (x**4)*dw12dv 
        t3 = -((y**6)*dw40dv + 3.*(y**4)*(x**2)*dw31dv + 3.*(y**2)*(x**4)*dw22dv + (x**6)*dw13dv)/3.
        s1 = -2.*x*W11
        s2 = 4.*(y**2)*x*W21 + 4.*(x**3)*W12
        s3 = -(2.*(y**4)*x*W31 + 4.*(y**2)*(x**3)*W22 + 2.*(x**5)*W13)
    
        dPhidx = (t0 + t1 + t2 + t3)*dvdx + (s1 + s2 + s3)
    
    # calculate dPhidy
        s1 = -2.*y*W20
        s2 = 4.*(y**3)*W30 + 4.*y*(x**2)*W21 
        s3 = -(2.*(y**5)*W40 + 4.*(y**3)*(x**2)*W31 + 2.*y*(x**4)*W22)
    
        dPhidy = (t0 + t1 + t2 + t3)*dvdy + (s1 + s2 + s3)
      
    r2 = (x**2 + y**2)
    Phix = Q*dPhidx - P*x/(r2+AA**2) + x
    
    Phiy = Q*dPhidy - P*y/(r2+AA**2) + y
        
    derivs =  (z2, z3, -2*z3 + Phix, 2*z2 + Phiy)
    return derivs   
    
# ---------------------------------------------------

""" phi(x,y) calculates the potential (x,y) - needed to check E_J """

def phi(x,y):  

    e = config.e
    a = 1/np.sqrt(1 - e**2)
    c = e*a
    AA = config.AA      # disk potential is 0.5 V^2 log(r^2 + AA^2)
    Mbar = config.Mbar  # units M_sun
    V = config.V        # units km/s
    eps = config.eps    # units kpc
    P = config.P        # P = 1,2 => CR at r = eps, 2eps
    Q = P*0.98*(Mbar/2e+10)*(5./eps)*((240/V)**2)
     
    # -----------------------------------

    def fb(v):                  # fn for solving for v via brentq
        g = (x**2)*(v**2)*(1-(v**2)) + ((y**2)+1)*(v**2)-1
        return g
        
    # -----------------------------------  
        
    if ((x/a)**2 + (y/c)**2 <= 1.):   # x,y within ellipsoid figure
        v = 1/a                        
    
    else:                             # x,y outside ellipsoid figure
        u = np.linspace(0,1,11)
        g = (x**2)*(u**2)*(1-(u**2)) + ((y**2)+1)*(u**2) -1 
        q = [i for i in range(0,10) if (g[i]*g[i+1] <= 0)][0] 
        aa = u[q]
        bb = u[q+1]
        v = brentq(fb,aa,bb)                             
        
    I0 = 2*v
    I1 = (v**3)/1.5
    I2 = (v**5)/2.5
    J = np.log((1.+v)/(1.-v))
    J0 = (2.*v/(1-v**2.) + J)/2.
    J1 = (2.*(v+v**3)/(1-v**2)**2 - J)/8.

    W10 = J
    W20 = J0 - W10
    W11 = W10 - I0
    W30 = J1 - W20
    W12 = W11 - I1
    W21 = W20 - W11
    W40 = (66.*v**5 - 80.*v**3 -15.*((1-v**2)**3)*J + 30.*v)/(48.*((1-v**2)**3))
    W22 = W21 - W12
    W31 = W30 - W21
    W13 = W12 - I2
    
    # calculate potential
    
   
    t1 = W10/3.
    t2 = - (W20*y**2 + W11*x**2)
    t3 = W30*y**4 + 2.*W21*(y**2)*(x**2) + W12*(x**4)
    t4 = -(W40*y**6 + 3.*W31*(y**4)*(x**2) + 3.*W22*(y**2)*(x**4) + W13*x**6)/3.
    Phi_b = t1 + t2 +t3 + t4    
    r2 = (x**2 + y**2)
    Phi = Q*Phi_b - P*0.5*np.log(r2 + AA**2) + 0.5*(r2)
    return Phi      
    
    