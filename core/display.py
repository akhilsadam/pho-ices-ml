import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
backend = plt.get_backend()
import jpcm
plt.switch_backend(backend)

def ecran(net, test_X, test_y, criterion, report, figsize=(15,5), classification=True):
    fig, axs = plt.subplots(1,3,figsize=figsize)
    
    try:
        outputs = net(test_X)
        loss = criterion(outputs, test_y)

        out =  np.argmax(outputs.detach().cpu().numpy(),axis=-1)
        true = np.argmax(test_y.detach().cpu().numpy(),axis=-1)
        title = f'- Performance - \nLoss: {loss.item()}'
        if classification:
            err = (out == true).astype(np.int8)
            ar =  np.sum(err) / len(err)
            title += f'\n{100*(ar):.2f}% Accuracy, or {100*(1-ar):.2f}% Error'


        axs[2].set_title(title)
    except:
        print("Test failed")

    try:
        cnf = confusion_matrix(true, out)
        im = axs[2].imshow(np.log10(cnf),cmap=jpcm.get('fuyu'))
        fig.colorbar(im, label='power of 10', ax=axs[2])
    except:
        print('confusion matrix failed')

    if report is not None:
        [rl, sizes, mix, miloss, losses] = report
        axs[0].set_title('Rate Curves')
        axs[0].plot(rl/np.max(rl),label='learning_rate',linestyle='--',color=jpcm.maps.karakurenai)
        axs[0].plot(sizes/np.max(sizes),label='batch_size',color=jpcm.maps.sora_iro)
        axs[0].set_xlabel('epochs')
        axs[0].legend()

        axs[1].set_title('Evolution')
        if len(losses.shape) > 1:
            for i in range(losses.shape[1]):
                axs[1].plot(losses[:,i],label=f'epoch loss {i}',color=jpcm.maps.karakurenai)
                axs[1].scatter(mix,miloss[:, i],label=f'batch loss {i}', s=1,alpha=0.5, color=jpcm.maps.chigusa_iro)
        else:
            axs[1].plot(losses,label='epoch loss',color=jpcm.maps.karakurenai)
            axs[1].scatter(mix,miloss,label='batch loss', s=1,alpha=0.5, color=jpcm.maps.chigusa_iro)
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('loss')
        axs[1].legend()
    
    plt.show()
    plt.close()