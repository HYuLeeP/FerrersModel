import numpy as np

e = 0.2
a = 1/np.sqrt(1 - e**2)
c = e*a
AA = 0.5                                      # disk potential is 0.5 V^2 log(r^2 + AA^2)
Mbar = 2.0e+10                                # units M_sun
V = 240.                                      # units km/s
eps = 5.                                      # units kpc
# P = 2.                                      # omega_bar = 33.9  km/s/kpc
P = 1.44                                      # omega_bar = 40
Q = P*0.98*(Mbar/2e+10)*(5./eps)*((240/V)**2)