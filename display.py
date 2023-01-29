import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
backend = plt.get_backend()
import jpcm
plt.switch_backend(backend)

def ecran(net, test_X, test_y, criterion, report, figsize=(15,5)):
    
    outputs = net(test_X)
    loss = criterion(outputs, test_y)

    out =  np.argmax(outputs.detach().cpu().numpy(),axis=-1)
    true = np.argmax(test_y.detach().cpu().numpy(),axis=-1)
    err = (out == true).astype(np.int8)
    cnf = confusion_matrix(true, out)

    ar =  np.sum(err) / len(err)
    
    fig, axs = plt.subplots(1,3,figsize=figsize)
    
    axs[2].set_title(f'- Performance - \nLoss: {loss.item()}\n{100*(ar):.2f}% Accuracy, or {100*(1-ar):.2f}% Error')
    im = axs[2].imshow(np.log10(cnf),cmap=jpcm.get('fuyu'))
    fig.colorbar(im, label='power of 10', ax=axs[2])
    
    if report is not None:
        [rl, sizes, mix, miloss, losses] = report
        axs[0].set_title('Rate Curves')
        axs[0].plot(rl/np.max(rl),label='learning_rate',linestyle='--',color=jpcm.maps.karakurenai)
        axs[0].plot(sizes/np.max(sizes),label='batch_size',color=jpcm.maps.sora_iro)
        axs[0].set_xlabel('epochs')
        axs[0].legend()

        axs[1].set_title('Evolution')
        axs[1].plot(losses,label='epoch loss',color=jpcm.maps.karakurenai)
        axs[1].scatter(mix,miloss,label='batch loss', s=1,alpha=0.5, color=jpcm.maps.chigusa_iro)
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('loss')
        axs[1].legend()
    
    plt.show()
    plt.close()