# This is the attempt to put the ferrers model + log potential into a class notation
import numpy as np
import kcf_L4_mod
from scipy.integrate import odeint
import config
import matplotlib.pyplot as plt

class InitialCondition:
    """
    The 4D initial condition defined for an orbit for numerical integration in the corotating frame

    the initial condition in phase space z0 = (x0,y0,vx0,vy0) can be defined directly or be inferred from EJ, position, and direction of motion

    ----------------------------------------------------------------
    Attributes:
    ej:         Jacobi Integral 
    x0:         initial x-Position (default:0)
    y0:         initial y-Position (default:0)
    direction:  initial direction of motion in degrees (default:0)
    z0:         Initial 4D phase space condition
    phi:        Effective potential (default:ferrers)
    dphi:       Potential gradient (vx, vy, dPhi/dx, dPhi/dy)
    ----------------------------------------------------------------
    Methods:    
    InitFromEj():   infer z0 from EJ, position, and direction of motion
    EjFromInit():   calculate EJ,x0,y0,and direction from z0
    """
    
    def __init__(self,ej:float=None,x0:float=0, y0:float=0, direction:float=0,z0=None,phi=kcf_L4_mod.phi,dphi=kcf_L4_mod.dphi) -> None:
        # we can define by y0/ej
        self.ej = ej 
        self.direction = direction #theta from right in degrees
        self.x0 = x0
        self.y0 = y0

        # we can also define by the 4D inital condition
        self.z0 : np.array = np.array(z0)

        # include phi for conversion
        self.phi = phi
        self.dphi = dphi
    
    def __str__(self) -> str:
        return f"EJ = {self.ej:.4f}, y0 = {self.y0:.2f}, x0 = {self.x0:.2f}, direction = {self.direction:.2f} deg \n[x0, y0, vx0, vy0] = {self.z0}"
    
    def InitFromEj(self)-> None:
        theta = self.direction/180*np.pi
        # TO check: phi is positive & inverted to usual convention? i.e -phi=EJ at L4
        vx0 = np.sqrt( 2.* (self.ej + self.phi(self.x0,self.y0)))*np.cos(theta)
        vy0 = np.sqrt( 2.* (self.ej + self.phi(self.x0,self.y0)))*np.sin(theta)
        self.z0 = [self.x0,self.y0,vx0,vy0]
        # return self.z0
    
    def EjFromInit(self)-> None:
        self.x0,self.y0,vx0,vy0 = self.z0
        self.ej = 0.5 * (vx0**2 + vy0**2) - self.phi(self.x0,self.y0)

        # treat vx0=0 differently
        if vx0 == 0:
            if vy0 == 0:
                temp=0
            else:
                temp = vy0*np.inf

        else:
            temp = vy0/vx0
        self.direction = np.arctan(temp)/np.pi*180

class Orbit:
    """
    An Orbit with the corresponding Initial Condition

    ----------------------------------------------------------------
    Attributes:
    init:   InitialCondition
    phi:    Effective Potential, used for Jacobi Integral Tracking
    dphi:   Potential gradient used to solve equations of motion
    tmax:   Maximum time for integration (default: 500 ~ 12 Gyr)
    rhot:   Density in the time space
    
    orbit:  (x,y,vx,vy) Integrated Orbit, obtain with IntOrbit()
    ----------------------------------------------------------------
    Methods:
    TrackJac(): Used to track the change in EJ throughout the orbit (Convservation of EJ)
    IntOrbit(): integrate the orbit (x,y,vx,vy)
                return: (x,y,vx,vy) numpy arrays
    """
    orbit = None
    def __init__(self, init, tmax=500, rhot=200) -> None:
        self.init : InitialCondition = init
        self.phi = init.phi
        self.dphi = init.dphi
        self.tmax = tmax
        self.rhot=rhot
        
    
    def __str__(self) -> str:
        output=f"Orbit with Initial condition:\n"+str(self.init) #inital condition
        output+=f"\nPotential model: {self.phi.__module__}.{self.phi.__name__}" #potential used
        output+=f"\nPotential gradient: {self.dphi.__module__}.{self.dphi.__name__}" #potential gradient used
        output+=f"\ntmax = {self.tmax}" #tmax for gradient
        return output

    def TrackJac(self):
        x,y,vx,vy=self.orbit
        F=np.zeros_like(x)
        for i in range(len(F)):
            F[i]= self.phi(x[i],y[i])
        Jac = 0.5*(vx**2) + 0.5*(vy**2) - F
        print(f"Jac_min, Jac_max = {np.min(Jac):.6f}, {np.max(Jac):.6f} \nDelta_Jac = {np.max(Jac)-np.min(Jac):.2e}")

    def IntOrbit(self) -> np.array:
        if self.init.z0 is None:
            self.init.InitFromEj()
        z0=self.init.z0

        self.nsamples : int = round(self.rhot*self.tmax)
        ts = np.linspace(0,self.tmax,self.nsamples)

        self.orbit = odeint(self.dphi, z0, ts).T
        return self.orbit


class PlotFerrers():
    """
    PlotFerrers: A PlotFerrers Class that specifically helps plotting in the ferrers model

    ----------------------------------------------------------------
    Attributes:
    orbit:      Orbit Class
    init:       InitialCondition Class
    color:      default color used for orbit plotting
    fontsize:   fontsize of labels
    ----------------------------------------------------------------
    Methods:
    GetOrbit():                                             Get the orbit if only the inital condition is provided
    PlotBar(ax:plt,zorder,markersize):                      Plots the bar and Lagrange points at the XY plot
    PlotZVC(axmax,ax):                                      Plot the ZVC for the corresponding EJ
    LabelXY(ax):                                            Label The xy-axes and title for the XY plot.
    PlotXY(ax,linewidth,axis,auto):                         Plot the orbit on the XY plane
    PlotSNd(ax,R,markersize,linewidth,zorder):              Plot the Solar Neighbourhood on the XY plot
    PlotM2K(ax,neighbour_condition,color,zorder,auto):      Plotting the lzvr kinematics for the orbit under certain neighbourhood condition
    PlotSOS(ax,markersize,color,style,to_right,axmax,auto): 
    """
    def __init__(self,orbit:Orbit=None,init:InitialCondition=None,
                 color="#006666",fontsize=12) -> None:
        self.orbit=orbit
        if init is None:
            self.init = orbit.init
        if init.z0 is None:
            init.InitFromEj()
        self.init=init
        if self.orbit is None:
            self.orbit = Orbit(init=self.init)
        self.color=color
        self.fontsize=fontsize

    def __str__(self) -> str:
        return str(self.init)+f"\nThe orbit is plotted in {self.color}"

    def GetOrbit(self):
        """
        GetOrbit: Get the orbit if only the inital condition is provided
        """
        
        if self.orbit.orbit is not None:
            tf=str(input("an orbit is given, reintegrate from inital condition? (y/n)"))
            if tf=='n':
                pass
            else:
                self.orbit = Orbit(init=self.init)
        self.__orbit=self.orbit.IntOrbit()
    
    def PlotBar(self,ax:plt.axes,zorder=10,markersize=3):
        """
        PlotBar: Plots the bar and Lagrange points at the XY plot

        Args:
            ax (plt.axes):              figure to plot
            zorder (int, optional):     Priority of the bar in the plotting. Defaults to 10.
            markersize (int, optional): size of the Lagrange points. Defaults to 3.
        """
        e = config.e
        a = 1/np.sqrt(1 - e**2)
        c = e*a
        # draw the Ferrers bar
        theta = np.linspace(0, 2*np.pi, 100)
        xf = a*np.sin(theta)
        yf = c*np.cos(theta)
        
        ax.plot(xf,yf, color='0.85')
    #   plt.fill(xf,yf,'gray')
        ax.fill(xf,yf,color='0.85')


    # plot L3, L4 for P = 1.44

        ax.plot(0.,1.24692,'bo', markersize = markersize,zorder=zorder)
        ax.plot(1.29546,0.,'bo', markersize = markersize,zorder=zorder)
        ax.plot(0.,-1.24692,'bo', markersize = markersize,zorder=zorder)
        ax.plot(-1.29546,0.,'bo', markersize = markersize,zorder=zorder)
        ax.plot(0.,0.,'bo', markersize = markersize,zorder=zorder)
        # ax.axis([-axmax,axmax,-axmax,axmax])

    def PlotZVC(self,axmax:float,ax:plt.axes):
        """
        PlotZVC: Plot the ZVC for the corresponding EJ

        Args:
            axmax (float): maximum of the meshgrid for the ZVC to be calculated
            ax (plt.axes): figure to plot
        """
        x = np.linspace(-axmax,axmax,201) + 0.00001  
        y = np.linspace(-axmax,axmax,201)

        ll = len(x)
        Phi = np.zeros((ll,ll))
        for i in np.arange(ll):
            for j in np.arange(ll):
                Phi[i,j] = self.init.phi(y[i],x[j])

        ax.contour(x,y,Phi.T,[-self.init.ej], colors=['blue'],linestyles='dashed')

    def LabelXY(self,ax:plt.axes):
        """
        LabelXY: Label The xy-axes and title for the XY plot.

        Args:
            ax (plt.axes): figure to plot
        """
        ax.set_xlabel("x (5 kpc)",fontsize=self.fontsize)
        ax.set_ylabel("y (5 kpc)",fontsize=self.fontsize)
        ax.set_title(f"x0, y0 = {self.init.x0:.2f}, {self.init.y0:.2f}; EJ = {self.init.ej}",fontsize=self.fontsize)

    def PlotXY(self,ax:plt.axes,linewidth=1,axis=[-2.5,2.5,-2.5,2.5],auto=True,label=None):
        """
        PlotXY: Plot the orbit on the XY plane

        Args:
            ax (plt.axes): figure to plot
            linewidth (int, optional): linewidth of the orbit. Defaults to 1.
            axis (float, optional): axis max/min for the XY plane. Defaults to [-2.5,2.5,-2.5,2.5].
            auto (bool, optional): Whether we want to plot the bar,ZVC,and label XY automatically. Defaults to True.
        """

        x,y,_,_=self.__orbit
        ax.plot(x,y,'-',color=self.color,zorder=5,linewidth=linewidth,label=label)
        ax.plot(self.init.x0,self.init.y0,'ro',zorder=10)

        ax.axis(axis)
        ax.axis('equal')


        if auto:
            self.PlotBar(ax=ax)
            self.PlotZVC(axmax=max(axis),ax=ax)
            self.LabelXY(ax=ax)

    def PlotSNd(self,ax:plt.axes,R:float=1/5,markersize:float=20,linewidth:float=1,zorder:int=12):
        """
        PlotSNd: Plot the Solar Neighbourhood on the XY plot

        Args:
            ax (plt.axes):                  figure to plot
            R (float, optional):            Radius of the neighbourhood (not the distance from galactic centre to sun). Defaults to 1/5.
            markersize (float, optional):   size of the sun. Defaults to 20.
            linewidth (float, optional):    width of the orbit passing through SNd. Defaults to 1.
            zorder (int, optional):         zorder of the SNd. Defaults to 12.
        """
        x,y,_,_=self.__orbit
        cdt=((x-1.486)**2+(y-.693)**2 < R**2)
        ax.plot(1.486,0.693,"r*",markersize=markersize,zorder=100)
        ax.plot(x[cdt],y[cdt],'.',color='cyan',linewidth=linewidth,zorder=zorder)

    def PlotM2K(self,ax:plt.axes,neighbour_condition=None,color:str='blue',zorder:int=100,auto:bool=True,label=None):
        """
        PlotM2K: Plotting the lzvr kinematics for the orbit under certain neighbourhood condition

        Args:
            ax (plt.axes): figure to plot
            neighbour_condition (cdt, optional):    neighbourhood condition. Defaults to SNd.
            color (str, optional):                  color of the kinematics. Defaults to 'blue'.
            zorder (int, optional):                 zorder of the kinematics. Defaults to 100.
            auto (bool, optional):                  True/False to automatically plot axis xlabel etc. Defaults to True.

        Returns:
            Lz_Nd,Vr_Nd:    np arrays of neighbourhood kinematics
        """
        x,y,vx,vy = self.__orbit
        Lz = x**2 + y**2 - x*vy + y*vx #(omega*R+vphi*R))
        r = np.sqrt(x**2+y**2)
        vr = (x*vx + y*vy)/r
        vr_ph=vr*5*40
        Lz_ph=Lz*5**2*40

        # cdt_SNd = ((x-xc)**2+(y-yc)**2 < 1/25)
        if neighbour_condition is None:
            neighbour_condition=((x-1.486)**2+(y-.693)**2 < 1/25)
        Lz_Nd = Lz_ph[neighbour_condition] # SNd
        vr_Nd = vr_ph[neighbour_condition]
        ax.plot(vr_Nd,Lz_Nd,'o',color=color,zorder=zorder,label=label)
        
        if auto:
            ax.axis([-150,150,1000,2500])
            ax.set_xlabel(r'$V_{R}$ (km/s)',fontsize=self.fontsize)
            ax.set_ylabel(r'$L_Z$ (kpc $\cdot$ km/s)',fontsize=self.fontsize)

        return Lz_Nd,vr_Nd
    
    def PlotSOS(self,ax:plt.axes,markersize:float=5,color:str="#006666",style:str='.',to_right:bool=True,axis:list=[-2.5,2.5,-2.5,2.5],auto:bool=True,label=None):
        """
        PlotSOS: Plot the Surface of Section of the orbit

        Args:
            ax (plt.axes):                  figure to plot on
            markersize (float, optional):   Size of the points on the SOS. Defaults to 5.
            color (str, optional):          color of the points on the SOS. Defaults to "#006666".
            style (str, optional):          Style of the ponits on the SOS. Defaults to '.'.
            to_right (bool, optional):      define on the right-crossing. Defaults to True.
            axis (list, optional):          list of the axes maxima. Defaults to [-2.5,2.5,-2.5,2.5].
            auto (bool, optional):          True/False of plotting bar/zvc/labels etc. Defaults to True.

        Returns:
            xs,ys: the points on the surface of section
        """
        x,y,vx,vy=self.__orbit
        q = np.where(vy[0:self.orbit.nsamples-2]*vy[1:self.orbit.nsamples-1] < 0.)[0]

        xs = np.zeros(len(q))
        ys = np.zeros(len(q))
        vxs = np.zeros(len(q))
        for k in np.arange(len(q)-1):
            dq = np.abs(vy[q[k]]) / (np.abs(vy[q[k]]) + np.abs(vy[q[k]+1]))
            xs[k] = x[q[k]] + dq*(x[q[k]+1] - x[q[k]])
            ys[k] = y[q[k]] + dq*(y[q[k]+1] - y[q[k]])
            vxs[k] = vx[q[k]] + dq*(vx[q[k]+1] - vx[q[k]])

        if to_right:
            # plot xs,ys if vxs >= 0
            c = (vxs >= 0.)           ### change to <= 0 as needed ####
        else:
            c = (vxs <= 0.)
        xs=xs[c][:-1]
        ys=ys[c][:-1]

        ax.plot(xs,ys,marker=style, markersize = markersize,linestyle="None",color=color,mec='none',label=label)     # was 0.5

        if auto:
            self.PlotBar(ax=ax)
            self.PlotZVC(axmax=max(axis),ax=ax)
            ax.set_xlabel('xs',fontsize=self.fontsize)
            ax.set_ylabel('ys',fontsize=self.fontsize)
            ax.set_title('EJ = {:.4f} '.format(self.init.ej),fontsize=self.fontsize)
            ax.axis(axis)
        
        return xs,ys

