from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from IPython.display import display

class PlotGrid:
    """
    A Grid of subplots for plotting

    ----------------------------------------------------------------
    Attributes:
    figsize:tuple:      vertical,horizontal size of the figure (default(4,6))
    gridshape:tuple:    number of rows,columns of the grid  (default (1,1))
    height_ratios, width_ratios:    height and width ratios of the grids(e.g [1,1,3.5])
    hspace, wspace:     vertical and horizontal space between subplots  

    xlabel,ylabel,title: labeling for the whole plot

    axs:    list of np.axes included in the Grid, obtain with GetAXS()
    ----------------------------------------------------------------
    Methods:
    FigLabel(x_position,y_position,title_position,Mfontsize):   
            Give labels and title for the whole Figure
    GetAXS(axshape,labelit):
            Create axes on the Grid


    """
    xlabel : str = 'x'
    ylabel : str = 'y'
    title : str = 'Title'

    def __init__(self,figsize:tuple=None, gridshape:tuple=None,
                 height_ratios:list=None, width_ratios:list=None, 
                 hspace:float=None, wspace:float=None) -> None:
        # We will adopt mxn for figsize/gridshape where m is vertical height and n is width in a similar sense to matrices
        if figsize is None:
            figsize = (4,6)
        self.figsize = figsize
        if gridshape is None:
            gridshape = (1,1)
        self.gridshape = gridshape
        self.height_ratios = height_ratios
        self.width_ratios = width_ratios
        self.hspace = hspace
        self.wspace = wspace
    
    def __str__(self) -> str:
        msg = f"A subplot grid with"
        msg += f"\nfiguresize = {self.figsize}"
        msg += f"\ngridshape  = {self.gridshape}"
        msg += f"\nh_ratio, w_ratio = {self.height_ratios}, {self.width_ratios}"
        msg += f"\nhspace, wspace = {self.hspace}, {self.wspace}"
        if self.axs is None:
            msg += f"\n------------\nNo subplots are included."
        else:
            msg += f"\n------------\nThe grid has {len(self.axs)} subplots."
        self.fig.tight_layout()
        # display(self.fig)
        return msg

    def FigLabel(self,x_position=0.00,y_position=0.00,title_position=1,Mfontsize=14):
        """
        Give labels and title for the whole Figure
        Inputs:
            x_position=0.00:    position at bottom of xlabel
            y_position=0.00:    position at left of ylabel
            title_position=1:   position at top of title
            Mfontsize=14:       fontsize of the title and labels
        """
        self.fig.text(0.5, x_position, self.xlabel, ha='center', va='center', fontsize=Mfontsize)
        self.fig.text(y_position, 0.5, self.ylabel, ha='center', va='center', rotation='vertical',fontsize=Mfontsize)
        self.fig.text(0.5, title_position, self.title, ha='center', va='center', rotation='horizontal',fontsize=Mfontsize)

    def GetAXS(self,axshape=None,labelit=False):
        """
        Create axes on the Grid
        Inputs:
            axshape:    list of slices gs to create axes [[a,b,c,d]] ==== gs[a:b,c:d]; a==b and/or c==d for gs[:,:]. See Note below
            labelit:    True/False for creating the axes label and title for the whole figure.
        Outputs:
            axs:    list of plt.axes for plotting
        
        Note on defining axshape:
        axshape is a list of lists;
        each list in axshape is defined for an ax wanted,
        each list have four integers each for [a,b,c,d] ==== gs[a:b,c:d]
        put a==b and/or c==d if want gs[:,:]
        """

        figsize=self.figsize
        self.fig=plt.figure(figsize=(figsize[1],figsize[0]))
        gridshape=self.gridshape
        gs = GridSpec(gridshape[0], gridshape[1], 
                      height_ratios=self.height_ratios, 
                      width_ratios=self.width_ratios, 
                      hspace=self.hspace, wspace=self.wspace)

        axs=[]
        if axshape is None:
            for g in gs:
                ax=self.fig.add_subplot(g)
                ax.tick_params(axis='both', direction='in',bottom=True, top=True, left=True, right=True,zorder=100)
                axs.append(ax)
                
        else:
            for AXS in axshape:
                temp1 = slice(AXS[0],AXS[1])
                temp2 = slice(AXS[2],AXS[3])
                if AXS[0]==AXS[1]:
                    temp1 = slice(0,self.gridshape[0])
                if AXS[2]==AXS[3]:
                    temp2 = slice(0,self.gridshape[1])
                # print(gs[temp1,temp2])
                ax=self.fig.add_subplot(gs[temp1,temp2])
                ax.tick_params(axis='both', direction='in',bottom=True, top=True, left=True, right=True,zorder=100)
                axs.append(ax)
        self.axs=axs
        if labelit:
            self.FigLabel()
        return axs