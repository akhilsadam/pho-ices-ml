import numpy as np
import sakuzu as skz
from jpcm import maps
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt

cs = [maps.gunjo_iro, maps.sakuranezumi, maps.murasaki]

def init():
    plt.figure(figsize=(4,4))

def export(path):
    plt.legend()
    plt.margins(0.001)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')

#https://tex.stackexchange.com/questions/16207/image-from-includegraphics-showing-up-in-wrong-location

def fda(path):
    d = skz.Diagram(10,10,c=(1,1,1),ec=(0,0,0))
    # y = d.block(.05,.6,0.1,0.1,label=r'$\mathbf{y_j}$')
    x = d.block(.05,.59,0.05,0.05, label=r'$\mathbf{x}$')
    y = d.block(.05,.225,0.04,0.04, label=r'$\mathbf{y}$')
    arch = d.block(.55,.5,0.8,0.3, shape='Rectangle',label=r'Architecture', text_y=1.05, ec=(0,0,0,1), c=(1/255)*np.array([27,41,75,125]))
    train = d.block(.55,.225,0.8,0.04, shape='Rectangle',label=r'Training Scaffold', fs=120, text_y=-.5, ec=(0,0,0,1), c=(1/255)*np.array([27,41,75,125]))
    n = 3
    lprev = x
    for i in range(n):
        l = arch.block(.05+(.9*i+0.3)/n,.8,0.8/n,0.25, shape='Rectangle',label=f'Localizer {i+1}',c=maps.rurikon,)
        lin = arch.block(.16+(.9*i+0.3)/n,.4,0.8/n,0.25, shape='Rectangle',label=f'Regressor {i+1}',c=maps.azuki_iro)
        if i>0:
            p = arch.block(.16+(.9*i+0.3)/n,.15,0.1,0.1,shape='Circle',label=r'$+$',c=maps.karakurenai, ec=maps.kokushoku, lw=2)
            neg = train.block(-.1+(.9*i+0.3)/n,.5,0.5,0.5,shape='Circle',label=r'$-$',c=maps.gunjo_iro, ec=maps.kokushoku, lw=2)
            d.arrow(lin,p,bent=False)
            d.arrow(yihat,neg,bent=True, start_face="right", end_face="top")
            if i==1:
                d.arrow(pprev,p,bent=True,start_face='bottom',end_face='left')
                d.arrow(nprev,neg,bent=False)
            else:
                d.arrow(pprev,p,bent=False)
                d.arrow(nprev,neg,bent=False)
            pprev = p
            nprev = neg
        else:
            pprev = lin
            nprev = y
        d.arrow(l,lin)
        d.arrow(lprev,l,bent=False)
        lprev = l
        yihat = arch.block(.10+(.9*i+0.3)/n,-.15,0.1,0.1,label=r'$\hat{y_'+str(i+1)+r'}$')
        d.arrow(lin,yihat,bent=False, start_face="bottom", start_shift=(-0.8*0.06,0))
        yi = train.block(.10+(.9*i+0.3)/n,1.5,0.05,0.05,label=r'$\mathbf{y_'+str(i+1)+r'}$')
        d.arrow(nprev,yi,bent=True, start_face="right", end_face="bottom", end_shift=(0,-0.01))

    
        
    
    y = arch.block(.95,-.15,0.1,0.1,label=r'$\mathbf{\hat{y}}$')
    d.arrow(pprev,y,bent=True, start_face="right", end_face="top")
    # note = d.block(0.2, 0.25, 0.3, 0.1, label='$E$ represents expectation:\nin this case a simple mean\nof the affine outputs', ec=maps.kokushoku)

    # q = lambda ax: ax.annotate('Nx', xy=(0.775, 0.55), xytext=(0.775, 0.575), xycoords='axes fraction', 
    #     fontsize=16, ha='center', va='center',
    #     arrowprops=dict(arrowstyle='-[, widthB=8.4, lengthB=1.25', lw=2))
    # ann = d.block(0.5, 0.5, 1.0, 1.0, item = q)

    d.render(filename=path, bbox_inches=Bbox([[0,1.5],[10,7]]))
    return 'Base FDA Architecture', 0.8

def fda2(path):
    d = skz.Diagram(10,10,c=(1,1,1),ec=(0,0,0))
    # y = d.block(.05,.6,0.1,0.1,label=r'$\mathbf{y_j}$')
    x = d.block(.05,.59,0.1,0.1,label=r'$\mathbf{x_0}$')
    y = d.block(.05,.225,0.04,0.04, label=r'$\mathbf{y}$')
    arch = d.block(.55,.5,0.8,0.3, shape='Rectangle',label=r'Architecture', text_y=1.05, ec=(0,0,0,1), c=(1/255)*np.array([27,41,75,125]))
    train = d.block(.55,.225,0.8,0.04, shape='Rectangle',label=r'Training Scaffold', fs=120, text_y=-.5, ec=(0,0,0,1), c=(1/255)*np.array([27,41,75,125]))
    n = 3
    lprev = x
    for i in range(n):
        l = arch.block(.05+(i+0.3)/n,.8,0.9/n,0.25, shape='Rectangle',label=f'Affine Localizer {i+1}\n'+r'$\vec{x_'+str(i+1)+r'}=\sigma(W_{'+str(i+1)+r'}\vec{x_{'+str(i)+r'}}+\vec{b_{'+str(i+1)+r'}})$',c=maps.rurikon)
        lin = arch.block(.16+(.9*i+0.3)/n,.4,0.8/n,0.25, shape='Rectangle',label=f'Linear Regressor {i+1}\n' + r'$\hat{\vec{a_'+str(i+1)+r'}}=V_{'+str(i+1)+r'}\vec{x_{'+str(i+1)+r'}}$',c=maps.azuki_iro)
        if i>0:
            p = arch.block(.16+(.9*i+0.3)/n,.15,0.1,0.1,shape='Circle',label=r'$+$',c=maps.karakurenai, ec=maps.kokushoku, lw=2)
            neg = train.block(-.1+(.9*i+0.3)/n,.5,0.5,0.5,shape='Circle',label=r'$-$',c=maps.gunjo_iro, ec=maps.kokushoku, lw=2)
            d.arrow(lin,p,bent=False)
            d.arrow(yihat,neg,bent=True, start_face="right", end_face="top")
            if i==1:
                d.arrow(pprev,p,bent=True,start_face='bottom',end_face='left')
                d.arrow(nprev,neg,bent=False)
            else:
                d.arrow(pprev,p,bent=False)
                d.arrow(nprev,neg,bent=False)
            pprev = p
            nprev = neg
        else:
            pprev = lin
            nprev = y
        d.arrow(l,lin)
        d.arrow(lprev,l,bent=False)
        lprev = l
        yihat = arch.block(.10+(.9*i+0.3)/n,-.15,0.1,0.1,label=r'$\hat{y_'+str(i+1)+r'}$')
        d.arrow(lin,yihat,bent=False, start_face="bottom", start_shift=(-0.8*0.06,0))
        yi = train.block(.10+(.9*i+0.3)/n,1.5,0.05,0.05,label=r'$\mathbf{y_'+str(i+1)+r'}$')
        d.arrow(nprev,yi,bent=True, start_face="right", end_face="bottom", end_shift=(0,-0.01))

    
        
    
    y = arch.block(.95,-.15,0.1,0.1,label=r'$\mathbf{\hat{y}}$')
    d.arrow(pprev,y,bent=True, start_face="right", end_face="top")
    
    d.render(filename=path, bbox_inches=Bbox([[0,1.5],[10,7]]))
    return 'Base FDA Architecture', 0.8


def ffasup(path):
    d = skz.Diagram(10,10,c=(1,1,1),ec=(0,0,0))
    y = d.block(.05,.6,0.1,0.1,label=r'$\mathbf{y_j}$')
    x = d.block(.05,.4,0.1,0.1,label=r'$\mathbf{x_i}$')
    plus = d.block(.15,.5,0.05,0.05,shape='Circle',label=r'$\bigoplus$',ec=maps.rurikon)
    d.arrow(x,plus)
    d.arrow(y,plus)
    n = 2
    for i in range(n):
        l=d.block(.35+i*.75/n,.5,0.5/n,0.07,label=f'Affine {i}\n'+r'$\vec{a_'+str(i)+r'}=\vec{x}W^T+\vec{b}$',c=maps.rurikon)
        if i==0:
            d.arrow(plus,l)
        else:
            d.arrow(norm,l,bent=False)
        norm=d.block(.5375+i*.75/n,.5,0.16/n,0.07,label='Norm',c=maps.murasaki)
        d.arrow(l,norm, bent=False)

    p2 = d.block(l.rcenter[0],l.rcenter[1]-0.1,0.1,0.05,label=r'$E[\|a_l\|^2]$',ec=maps.rurikon)
    sigm = d.block(l.rcenter[0],l.rcenter[1]-0.2,0.05,0.05,shape='Circle',label=r'$\sigma(x)$',ec=maps.rurikon)
    d.arrow(l,p2)
    d.arrow(p2,sigm)
    p = d.block(sigm.rcenter[0],sigm.rcenter[1]-0.1,0.05,0.05,shape='Circle',label=r'$\mathbf{\hat{p}(y_j \mid x_i)}$')
    d.arrow(sigm,p)

    note = d.block(0.2, 0.25, 0.3, 0.1, label='$E$ represents expectation:\nin this case a simple mean\nof the affine outputs', ec=maps.kokushoku)

    q = lambda ax: ax.annotate('Nx', xy=(0.775, 0.55), xytext=(0.775, 0.575), xycoords='axes fraction', 
        fontsize=16, ha='center', va='center',
        arrowprops=dict(arrowstyle='-[, widthB=8.4, lengthB=1.25', lw=2))
    ann = d.block(0.5, 0.5, 1.0, 1.0, item = q)

    d.render(filename=path, bbox_inches=Bbox([[0,1.5],[10,7]]))
    return 'Supervised FFA Architecture', 0.8

for th in [0,1,2]:
    exec(r"""
def lfunc"""+str(th)+r"""(path):
    # let x = (theta-g+) (want to minimize)
    init()
    th = float(path.split('.pd')[0][-1])
    x = np.linspace(0,10,1000)-th
    s = lambda y: 1/(1+np.exp(-y))
    l0 = np.log(s(x))
    l1 = -np.log(s(-x))
    plt.plot(x,l0,label=r'$L_0 :=\log \sigma([g_+-\theta])$', color=cs[0])
    plt.plot(x,l1,label=r'$L_1 :=-\log \sigma([\theta-g_+])$', color=cs[1])
    plt.plot(x, l0+l1, label=r'$L_0+L_1$; constant ascent', color=cs[2])
    plt.xlabel(r'$g_+-\theta$' + '  [want to maximize over training]   '+ r'$\theta=$'+str(th))
    plt.ylabel('Loss')
    export(path)
    return 'FFA Loss',0.32

def lfunc2"""+str(th)+r"""(path):
    # let x = (theta-g+) (want to minimize)
    init()
    th = float(path.split('.pd')[0][-1])
    n = 1200
    x = np.linspace(0,12,n)-th
    s = lambda y: 1/(1+np.exp(-y))
    t0 = np.cumsum(1/(1-s(x)))/n#(1+np.exp(x))/np.exp(x),(1+np.exp(-x))/np.exp(-x)
    t1 = np.cumsum(1/(1-s(-x)))/n
    t2 = (x+th)
    plt.plot(t0,x,label=r'$L_0$', color=cs[0])
    plt.plot(t1,x,label=r'$L_1$', color=cs[1])
    plt.plot(t2,x,label=r'$L_0+L_1$; constant ascent', color=cs[2])
    plt.xlim(0,10)
    plt.ylim(-th,10)
    plt.ylabel('position [want to maximize over training]')
    plt.xlabel('time taken')
    export(path)
    return 'FFA Time to Position',0.32

def lfunc3"""+str(th)+r"""(path):
    # let x = (theta-g+) (want to minimize)
    init()
    th = float(path.split('.pd')[0][-1])
    n = 1000
    x = np.linspace(0,10,n)-th
    s = lambda y: 1/(1+np.exp(-y))
    l0 = np.log(s(x))
    l1 = -np.log(s(-x))

    t0 = np.cumsum(1/(1-s(x)))/n#(1+np.exp(x))/np.exp(x),(1+np.exp(-x))/np.exp(-x)
    t1 = np.cumsum(1/(1-s(-x)))/n
    t2 = (x+th)

    plt.plot(t0,l0,label=r'$L_0$',color=cs[0])
    plt.plot(t1,l1,label=r'$L_1$',color=cs[1])
    plt.plot(t2,l0+l1, label=r'$L_0+L_1$; constant ascent',color=cs[2])
    plt.xlim(0,10)
    plt.ylabel('Loss')
    plt.xlabel('time taken')
    export(path)
    return 'FFA Loss over Time',0.32
""")