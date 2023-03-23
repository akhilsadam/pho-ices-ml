import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
backend = plt.get_backend()
import jpcm
plt.switch_backend(backend)
import torch

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def ecran(net, test_X, test_y, report=None, criterion=torch.nn.MSELoss(), figsize=(21,7), classification=False):
    
    fig, axs = plt.subplots(1,3 if classification else 2,figsize=figsize, width_ratios=[1,2,2] if classification else [1,2])
    plt.subplots_adjust(wspace=0.2)
    
    try:
        outputs = net(test_X)
        loss = criterion(outputs, test_y)

        out =  np.argmax(outputs.detach().cpu().numpy(),axis=-1)
        true = np.argmax(test_y.detach().cpu().numpy(),axis=-1)
        title = f'- Performance - \nLoss: {loss.item()}'
        t2 = 'Evolution'
        if classification:
            err = (out == true).astype(np.int8)
            ar =  np.sum(err) / len(err)
            title += f'\n{100*(ar):.2f}% Accuracy, or {100*(1-ar):.2f}% Error'
            
            try:
                axs[2].set_title(title)
                cnf = confusion_matrix(true, out)
                im = axs[2].imshow(np.log10(cnf),cmap=jpcm.get('fuyu'))
                fig.colorbar(im, label='power of 10', ax=axs[2])
            except:
                print('confusion matrix failed')
        else:
            t2 += f'\nTest Loss: {loss.item()}'
        
    except:
        print("Test failed")



    if report is not None:
        plot_test = False
        if len(report) == 5:
            [rl, sizes, mix, miloss, losses] = report
        elif len(report) == 6:
            [rl, sizes, mix, miloss, losses, lossnames] = report
        elif len(report) == 8:
            [rl, sizes, mix, miloss, losses, lossnames, testx,testlosses] = report
            plot_test = True
        else:
            raise('report is not the right length')           
            
        
        n = len(losses.shape)
        nc = round(1.5*n+0.4)
        cs = jpcm.get('sky').resampled(nc).colors[:n]
        
        
        axs[0].set_title('Rate Curves')
        p1, = axs[0].plot(rl,label='learning_rate',linestyle='--',color=jpcm.maps.karakurenai)
        ax = axs[0].twinx()
        ax.spines.right.set_position(('axes', 1.0))
        ax.set_ylabel('batch_size')
        p2, = ax.plot(sizes,label='batch_size',color=jpcm.maps.sora_iro)
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('learning rate')
        axs[0].legend(handles=[p1,p2])

        axs[1].set_title(t2)
        handles = []
        ax2 = axs[1].inset_axes([0.89, 0.63, 0.10, 0.18])
        ax3 = axs[1].inset_axes([0.89, 0.40, 0.10, 0.18])  
        
        if plot_test:
            ax = axs[1].twinx()
            ax.spines['left'].set_position(('axes', 0.0))
            make_patch_spines_invisible(ax)
            ax.spines['left'].set_visible(True)
            ax.yaxis.set_ticks_position('left')
            ax.yaxis.set_label_position('left')
            ax.set_ylabel('test loss')
            ax.set_yscale('log')
            handles.append(ax.plot(testx,testlosses,label='test loss',color=jpcm.maps.karakurenai,)[0])
            ax3.plot(testx,testlosses,label='test loss',color=jpcm.maps.karakurenai)
        
        if len(losses.shape) > 1:
            names = lossnames if lossnames is not None else [f'loss {i}' for i in range(losses.shape[1])]
            for i in range(losses.shape[1]):
                # reverse order
                i = losses.shape[1]-i-1
                ax = axs[1].twinx()
                ax.spines.right.set_position(('axes', 1.1**i))
                ax.set_ylabel(names[i])
                ax.set_yscale('log')
                handles.append(ax.plot(losses[:,i],label=f'train epoch: {names[i]}',linewidth=1,color=cs[i])[0])
                ax.scatter(mix,miloss[:, i],label=f'train batch: {names[i]}', s=1,alpha=0.5, color=cs[i])
                ax2.plot(losses[:,i],label=f'train epoch: {names[i]}',color=cs[i])
        else:
            ax = axs[1].twinx()
            ax.spines.right.set_position(('axes', 1.2))
            ax.set_ylabel('train loss')
            ax.set_yscale('log')
            handles.append(ax.plot(losses,label='train epoch',color=jpcm.maps.rurikon)[0])
            ax.scatter(mix,miloss,label='train batch', s=1,alpha=0.5, color=jpcm.maps.chigusa_iro)
            ax2.plot(losses,label='train epoch',color=jpcm.maps.rurikon)

            
        
        
        ax2.set_yticks([])    
        ax2.set_xticks([])
        ax2.set_xlabel('epochs')
        ax3.set_yticks([])    
        ax3.set_xticks([])
        axs[1].set_xlim(10, mix[-1])
        axs[1].yaxis.set_tick_params(labelleft=False) 
        axs[1].set_yticks([])
        axs[1].set_xlabel('epochs (skipping first 10)')
        axs[1].legend(handles = handles)

    plt.show()
    plt.close()