"""
swarmSimQt.py
 - version using perimeter test,
 - David's tidy-up of my onPerim(b), incorporated into 'perimeter' branch of the notebook 2020Mar4

Uses PyQt5-based animated display.
(1) The swarm moves, perimeter agents are shown red, inner agents black.
It's interactive -
(2) The display pane is scrollable and zoomable (scale-factor can be varied).
(3) Animation can be paused/restarted and sped up/slowed down.
(4) Moving the mouse within the display area prints logical (ie, in swarm
coordinate system) position on the console. y-axis points up.
(3) A mouse click anywhere in the display area causes swarm coordinates
and perimeter status the be displayed on console. If the position pointed at
is within cohesion range of 1 or more agent, the data for these are displayed;
otherwise data for ALL agents displayed.
(4) Axes are drawn crossing at logical (0,0), with grid lines at 100-pixel intervals.
With default pane dimensions of of 1000x1000px and scale factor of 50 this
this give grid line at intervals of 5 in swarm coords. Zoom is by factors of 10
giving grid interval = 5x10^n in logical (swarm) coordinates depending on zoom factor.
"""
import numpy as np

############################## swarm model stuff ###############################
# Define some useful array accessor constants
POS_X  = 0    # x-coordinates of agents position 
POS_Y  = 1    # y-coordinates of agents position
COH_X  = 2    # x-coordinates of cohesion vectors
COH_Y  = 3    # y-coordinates of cohesion vectors
REP_X  = 4    # x-coordinates of repulsion vectors
REP_Y  = 5    # y-coordinates of repulsion vectors
DIR_X  = 6    # x-coordinates of direction vectors
DIR_Y  = 7    # y-coordinates of direction vectors
RES_X  = 8    # x-coordinates of resultant vectors
RES_Y  = 9    # y-coordinates of resultant vectors
GOAL_X = 10   # x-coordinates of goals
GOAL_Y = 11   # y-coordinates of goals
CF     = 12   # cohesion field radii
RF     = 13   # repulsion field radii
KC     = 14   # cohesion vector scaling factor
KR     = 15   # repulsion vector scaling factor
KD     = 16   # direction vector scaling factor
PRM    = 17

N_ROWS = 18   # number of rows in array that models swarm state

def mk_rand_swarm(n, *, cf=4.0, rf=3.0, kc=1.0, kr=1.0, kd=0.0, goal=0.0, loc=0.0, grid=10, seed=None):
    '''
    create a 2-D array of N_ROWS attributes for n agents. 
    :param n:      number of agents
    :param cf:     all agents' cohesion field radius; dflt=4.0; heterogeneous fields not catered for here
    :param rf:     repulsion field radius of all agents; default 3.0
    :param kc:     weighting factor for cohesion component, default 1.0
    :param kr:     weighting factor for repulsion component, default 1.0
    :param kd:     weighting factor for direction component, default 0.0 (i.e. goal is ignored by default)
    :param goal:   all agents' goal loc; heterogeneous goals are allowed but not catered for here
    :param loc:    location of agent b_0 -- the focus of the swarm
    :param grid:   size of grid around b_0 in which all other agents will be placed initially at random
    '''
    b = np.empty((N_ROWS, n))                       #create a 2-D array, big enough for n agents
    prng = np.random.default_rng(seed)
    b[POS_X:POS_Y + 1,:] = (prng.random(size=2*n)*2*grid - grid + loc).reshape(2, n) # random placement
    b[POS_X:POS_Y + 1,0] = loc                      # b_0 placed at [loc, loc]       
    b[COH_X:COH_Y+1,:] = np.full((2,n), 0.0)        # cohesion vectors initially [0.0, 0.0]
    b[REP_X:REP_Y+1,:] = np.full((2,n), 0.0)        # repulsion vectors initially [0.0, 0.0]
    b[DIR_X:DIR_Y+1,:] = np.full((2,n), 0.0)        # direction vectors initially [0.0, 0.0]
    b[RES_X:RES_Y + 1,:] = np.full((2,n), 0.0)      # resultant vectors initially [0.0, 0.0]
    b[GOAL_X:GOAL_Y + 1,:] = np.full((2,n), goal)   # goal is at [goal, goal], default [0.0, 0.0]
    b[CF,:] = np.full(n, cf)                        # cohesion field of all agents set to cf
    b[RF,:] = np.full(n, rf)                        # repulsion field of all agents set to rf
    b[KC,:] = np.full(n, kc)                        # cohesion weight for all agents set to kc
    b[KR,:] = np.full(n, kr)                        # repulsion weight for all agents set to kr
    b[KD,:] = np.full(n, kd)                        # direction weight for all agents set to kd
    b[PRM,:] = np.full(n, False)                    # perimeter initially false
    return b

def mk_swarm(xs, ys, *, cf=4.0, rf=3.0, kc=1.0, kr=1.0, kd=0.0, goal=0.0, grid=10):
    '''
    create a 2-D array of N_ROWS attributes for len(xs) agents. 
    :param xs:      x-values of position of agents
    :param ys:      y-values of position of agents
    :param cf:      all agents' cohesion field radius; dflt=4.0; heterogeneous fields not catered for here
    :param rf:      repulsion field radius of all agents; default 3.0
    :param kc:      weighting factor for cohesion component, default 1.0
    :param kr:      weighting factor for repulsion component, default 1.0
    :param kd:      weighting factor for direction component, default 0.0 (i.e. goal is ignored by default)
    :param goal:    all agents' goal loc; heterogeneous goals are allowed but not catered for here
    :param loc:     location of agent b_0 -- the focus of the swarm
    :param grid:    size of grid around b_0 in which all other agents will be placed initially at random
    '''
    n = len(xs)
    assert len(ys) == n
    b = np.empty((N_ROWS, n))                       #create a 2-D array, big enough for n agents
    b[POS_X] = np.array(xs)                         # place agents as specified
    b[POS_Y] = np.array(ys)                         # place agents as specified       
    b[COH_X:COH_Y+1,:] = np.full((2,n), 0.0)        # cohesion vectors initially [0.0, 0.0]
    b[REP_X:REP_Y+1,:] = np.full((2,n), 0.0)        # repulsion vectors initially [0.0, 0.0]
    b[DIR_X:DIR_Y+1,:] = np.full((2,n), 0.0)        # direction vectors initially [0.0, 0.0]
    b[RES_X:RES_Y + 1,:] = np.full((2,n), 0.0)      # resultant vectors initially [0.0, 0.0]
    b[GOAL_X:GOAL_Y + 1,:] = np.full((2,n), goal)   # goal is at [goal, goal], default [0.0, 0.0]
    b[CF,:] = np.full(n, cf)                        # cohesion field of all agents set to cf
    b[RF,:] = np.full(n, rf)                        # repulsion field of all agents set to rf
    b[KC,:] = np.full(n, kc)                        # cohesion weight for all agents set to kc
    b[KR,:] = np.full(n, kr)                        # repulsion weight for all agents set to kr
    b[KD,:] = np.full(n, kd)                        # direction weight for all agents set to kd
    b[PRM,:] = np.full(n, False)                    # perimeter initially false
    return b

def onPerim(b, xv=None, yv=None, mag=None, coh_n=None):
    """
    Determines the perimeter status of all agents in swarm b
    
    :param b: a data structure representing the swarm
    :returns: a numpy array of bools, one element per agent set to True if agent is on perimeter, False otherwise
    """
    if xv is None:
        xv = np.subtract.outer(b[POS_X], b[POS_X])  # all pairs x-differences
        yv = np.subtract.outer(b[POS_Y], b[POS_Y])  # all pairs y-differences
        mag = np.hypot(xv, yv)                      # all pairs magnitudes
        coh_n = mag <= b[CF]                        # cohesion neighbours
        np.fill_diagonal(coh_n, False)              # no agent is a cohesion neighbour of itself
    else:
        assert(not (yv is None or mag is None or coh_n is None))
    
    ang = np.arctan2(yv, xv)                    # all pairs polar angles
    ang_coh = np.where(coh_n, ang, 10.0)        # polar angle for pairs of agents within coh range; ow dummy value 10

    def isAgentOnPerimeter(nba):
        """
        Determines the perimeter status of a single agent
        
        :param nba: array of neighbour angles for all cohesion neighbours of one agent
        :returns: True if perimeter condition is satisfied, otherwise False
        """
        nr = np.count_nonzero(nba<10)   # angles of coh neighbours are nba[i] for 0 <= i < nr
        if nr == 0:                     # agent has no nbrs ... 
            is_on_perimeter = True      # ... so perimeter condition satisfied immediately
        else:
            nbi = np.argsort(nba, axis=0).astype(int)[0:nr] # nbi indexes nba in ascending order of angle, losing dummy values
            adj = np.row_stack((nbi, np.roll(nbi,-1)))      # 2 x nr array of adjacent neighbours in which for 0 <= i < nr,
                                                            # adj[0, i] == nbi[i] and adj[1, i] == nbi[i + 1 % nr]

            def perimeterTest(p):           # the helper's helper
                """
                Tests if a pair of an agent's adjacent neighbours give the agent the 'perimeter-iness' property
                
                :param p: an array of shape (2,1): p[0] and p[1] are a pair of adjacent neighbours in polar angle order
                """
                if not coh_n[p[1],p[0]]:    # the adjacent pair are not cohesion neighbours of each other ...
                    result = True           # ... so the agent under consideration is on the perimeter
                else:
                    delta = nba[p[1]] - nba[p[0]]   # compute the angle between the adjacent neighbour pair
                    if delta < 0:
                        delta += np.pi * 2.0
                    result = (delta > np.pi)        # agent is on the perimeter if this is a reflex angle
                return result
            
            is_on_perimeter = np.any(np.apply_along_axis(perimeterTest, 0, adj))    
                # agent is on perimeter if any pair of its adjacent cohesion neighbours satisfies the perimeter test
        return is_on_perimeter

    return np.apply_along_axis(isAgentOnPerimeter, 0, ang_coh)


def d_step(b, *, scaling='linear', exp_rate=0.2, speed=0.05, with_perimeter=False, perimeter_directed=False):
    """
    Compute one step in the evolution of swarm `b`
    
    :param b: the array modelling the state of the swarm
    :param scaling: choose 'linear', 'quadratic', or 'exponential' scaling of repulsion vectors
    :param exp_rate: rate of scaling in 'exponential' case
    :param speed: the speed of each agent, i.e. the number of simulation distance units per simulation time unit (step)
    """
    xv = np.subtract.outer(b[POS_X], b[POS_X])  # all pairs x-differences
    yv = np.subtract.outer(b[POS_Y], b[POS_Y])  # all pairs y-differences

    # compute all pairwise vector magnitudes
    mag = np.hypot(xv, yv)        # all pairs magnitudes
    
    # compute the cohesion neighbours
    coh_n = mag <= b[CF]
    np.fill_diagonal(coh_n, False)     # no agent is a cohesion neighbour of itself
    nr_coh_n = np.sum(coh_n, axis = 0) # number of cohesion neighbours

    # compute the x-differences and y-differences for cohesion vectors
    xv_coh = np.where(coh_n, xv, 0.0)
    yv_coh = np.where(coh_n, yv, 0.0)

    # compute the cohesion vectors 
    b[COH_X] = xv_coh.sum(axis=0)                       # sum the x-differences 
    b[COH_Y] = yv_coh.sum(axis=0)                       # sum the y-differences
    b[COH_X:COH_Y+1] /= np.maximum(nr_coh_n, 1)         # divide by the number of cohesion neighbours

    # compute the repulsion neighbours
    rep_n = mag <= b[RF]
    np.fill_diagonal(rep_n, False)     # no agent is a repulsion neighbour of itself
    nr_rep_n = np.sum(rep_n, axis = 0) # number of repulsion neighbours

    # compute the x-differences and y-differences for repulsion vectors
    eps = np.finfo('float64').eps
    mag_nz = np.where(mag != 0, mag, eps)                                  
    if scaling == 'linear':                             # repulsion scaling factor
        rscalar = mag[rep_n] + (rep_n * -b[RF])[rep_n]              
    elif scaling == 'quadratic':
        rscalar = (rep_n * -b[RF])[rep_n] * (mag_nz[rep_n] ** (-2))
    elif scaling == 'exponential':
        rscalar = (rep_n * -b[RF])[rep_n] * (np.e ** (-mag[rep_n] * exp_rate))
    else:
        assert(False)
    xv_rep = np.full_like(xv, 0.)
    yv_rep = np.full_like(yv, 0.)
    xv_rep[rep_n] = xv[rep_n] / mag_nz[rep_n] * rscalar # scale the normalised x-values
    yv_rep[rep_n] = yv[rep_n] / mag_nz[rep_n] * rscalar # scale the normalised y-values

    # compute the resultant repulsion vectors 
    b[REP_X] = xv_rep.sum(axis=0)                       # sum the x-differences 
    b[REP_Y] = yv_rep.sum(axis=0)                       # sum the y-differences
    b[REP_X:REP_Y+1] /= np.maximum(nr_rep_n, 1)         # divide by the number of repulsion neighbours
    
    # compute the direction vectors
    b[DIR_X:DIR_Y+1] = b[GOAL_X:GOAL_Y+1] - b[POS_X:POS_Y+1]

    # compute the resultant of the cohesion, repulsion and direction vectors
    if perimeter_directed or with_perimeter:
        b[PRM] = onPerim(b, xv=xv, yv=yv, mag=mag, coh_n=coh_n)
    if perimeter_directed:
        b[RES_X:RES_Y+1] = b[KC] * b[COH_X:COH_Y+1] + b[KR] * b[REP_X:REP_Y+1] + b[PRM] * b[KD] * b[DIR_X:DIR_Y+1]
    else:
        b[RES_X:RES_Y+1] = b[KC] * b[COH_X:COH_Y+1] + b[KR] * b[REP_X:REP_Y+1] + b[KD] * b[DIR_X:DIR_Y+1]
                  
    # compute the resultant magnitudes and normalise the resultant
    mag_res = np.hypot(b[RES_X], b[RES_Y])
    mag_res = np.where(mag_res != 0, mag_res, eps)
    b[RES_X:RES_Y+1] /= mag_res

    # multiply resultant by factor for speed and update positions of agents
    b[RES_X:RES_Y+1] *= speed                           # distance units per time unit
    b[POS_X:POS_Y+1] += b[RES_X:RES_Y+1]                # update positions

    return mag, coh_n, rep_n   # helpful laterin calculation of metrics


############################## Qt Display ##################################

from PyQt5.QtWidgets import QWidget, QApplication, QScrollArea, QPushButton,\
                            QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPainter, QColor, QCursor
from PyQt5.QtCore import Qt, QTimer
import sys

pallette = [Qt.black, Qt.red, Qt.green, Qt.blue]

## Widget proving animated display - will go inside a scroll pane ##
class Display(QWidget):
   ## Initialise UI, model, timer ##  
  def __init__(self, data, width=800, height=600, scf=50.0, interval=64):
    super().__init__()
    self.initUI(width, height)
    self.dta = data
    self.scaleFact = scf
    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update) # QWidget.update() fires a paintEvent
    self.timer.start(interval)
    
  ## Initialise UI with preferred geometry, mouse tracking
  def initUI(self, width, height):    
    self.setGeometry(50, 50, width, height)
    self.setWindowTitle('Swarm Display')
    self.setMouseTracking(True)
    self.setCursor(QCursor(Qt.CrossCursor))
    self.show()

  # Display position pointed at my mouse, in swarm coordinates
  def mouseMoveEvent(self, event):
    ax = (event.x()/self.size().width() - 0.5) * self.scaleFact    
    ay = -(event.y()/self. size().height() - 0.5) * self.scaleFact    
    print("({0:.2f},{1:.2f})  ".format(ax,ay), end="\r")

  # If mouse clicked while animation paused, display data of agent(s) in range,
  # or all agents, if none in range
  def mousePressEvent(self, evt):
    if self.timer.isActive():
      return
    # swarm coords pointed at: 
    ax = (evt.x()/self.size().width() - 0.5) * self.scaleFact    
    ay = -(evt.y()/self. size().height() - 0.5) * self.scaleFact 
    # agents within range:
    inrange = np.hypot(self.dta[POS_X] - ax, self.dta[POS_Y] - ay) < self.dta[CF]
    if np.count_nonzero(inrange) == 0:
      print("All agents:    ")
      print("x:", self.dta[POS_X].round(2))
      print("y:", self.dta[POS_Y].round(2))
      print("p:", self.dta[PRM])
    else:
      print("Agents near ({0:.2f},{1:.2f}):".format(ax,ay))
      print("x:", np.extract(inrange, self.dta[POS_X]).round(2))
      print("y:", np.extract(inrange, self.dta[POS_Y]).round(2))
      print("p:", np.extract(inrange, self.dta[PRM]))

  def paintEvent(self, event):
    width = self.size().width()
    height = self.size().height()
    lh = self.dta.shape[1]
    #d_step(self.dta, scaling='quadratic')
    d_step(self.dta, scaling='quadratic', with_perimeter=True)
    #d_step(self.dta, scaling='quadratic', perimeter_directed=True)
    clrs = np.where(self.dta[PRM],1,0) 
    qp = QPainter()
    qp.begin(self)
    qp.setPen(Qt.cyan)
    for i in range(0, int(width/2), int(width/10)):
      qp.drawLine(width/2+i,0,width/2+i,height)
      qp.drawLine(width/2-i,0,width/2-i,height)
    for i in range(0, int(height/2), int(height/10)):
      qp.drawLine(0,height/2+i,width,height/2+i)
      qp.drawLine(0,height/2-i,width,height/2-i)
    qp.setPen(Qt.blue)
    qp.drawLine(0,height/2,width,height/2)
    qp.drawLine(width/2,0,width/2,height)
    for i in range(lh):
      gx = (self.dta[POS_X,i]/self.scaleFact + 0.5)*width
      gy = (-self.dta[POS_Y,i]/self.scaleFact + 0.5)*height
      qp.setPen(pallette[clrs[i]])
      qp.drawEllipse(gx-2, gy-2, 4, 4)   
    qp.end()    

  ## Timer control methods
  def faster(self):
    i = self.timer.interval()
    if i > 64:
      i //= 2
      self.timer.setInterval(i)

  def slower(self):
    i = self.timer.interval()
    if i < 1024:
      i *= 2
      self.timer.setInterval(i)

  def stopStart(self):
    if self.timer.isActive():
      self.timer.stop()
    else:
      self.timer.start()

  ## Zoom control methods
  def zoomOut(self):
    self.scaleFact *= 10

  def zoomIn(self):
    self.scaleFact /= 10
## 
## end of Display class

## 'Main window': contains timer control buttons in a panel sitting
##   above a scroll pane containing a Circles instance.  ###########
class Window(QWidget):
  def __init__(self, data):
    super().__init__()
    self.initUI(data)

  def initUI(self, data):
    pBtn = QPushButton("Pause/Resume")
    fBtn = QPushButton("Faster")
    sBtn = QPushButton("Slower")
    oBtn = QPushButton("Zoom out")
    iBtn = QPushButton("Zoom in")  
    hbox = QHBoxLayout()           ## these buttons laid out horizontally,
    hbox.addStretch(1)             ## centred by means of a stretch at each end
    hbox.addWidget(pBtn)
    hbox.addWidget(fBtn)
    hbox.addWidget(sBtn)
    hbox.addWidget(oBtn)
    hbox.addWidget(iBtn)
    hbox.addStretch(1)
    
    dsp = Display(data, width=1000, height=1000)  ## make a Display instance
    scrollArea = QScrollArea()                    ## and a scroll pain;
    scrollArea.setWidget(dsp)                 ## put the former in the latter 

    vbox = QVBoxLayout()    ## Lay the button panel out on top of the scrollArea
    vbox.addLayout(hbox)
    vbox.addWidget(scrollArea)
    self.setLayout(vbox)

    self.setGeometry(50, 50, 700, 700) # Initally 700x700 pixels with 50 px offsets

    pBtn.clicked.connect(dsp.stopStart) # Register handlers: 
    fBtn.clicked.connect(dsp.faster)    #   - methods in Display instance
    sBtn.clicked.connect(dsp.slower)
    oBtn.clicked.connect(dsp.zoomOut)
    iBtn.clicked.connect(dsp.zoomIn)
##
## End Window class

     
################################ main line ########################################

# b = mk_rand_swarm(12, cf=4.0, kr=5.0, grid=4.0)
# b = mk_rand_swarm(72, cf=4.0, kr=5.0, grid=4.0)
# b = mk_rand_swarm(200, rf=7.0, cf=8.0, kr=30.0, grid=20.0)
# b = mk_rand_swarm(200, rf=7.0, cf=8.0, kr=30.0, grid=10.0)
# b = mk_rand_swarm(200, rf=7.0, cf=8.0, kr=15.0, kd=1.0, grid=20.0)
# b = mk_rand_swarm(200, rf=7.0, cf=8.0, kr=100.0, grid=10.0)
b = mk_rand_swarm(400, loc=-7.5, cf=4.0, kd=1.0, kr=30.0, kc=1.0, grid=20.0, goal=20.0) # (rf=3.0)

app = QApplication(sys.argv)
win = Window(b)
win.show()
sys.exit(app.exec_())

