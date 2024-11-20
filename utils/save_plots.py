import torch, os
import matplotlib.pyplot as plt
import torch.nn.functional as F


def plot_reflectivity(X, X0, Xk, args, epoch):
    Xk = Xk.detach().cpu()
    X = X.detach().cpu()
    X0 = X0.detach().cpu()

    # plot 2D
    i = 0
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(X0[i, 0], cmap='gray')
    plt.title('Observed Trace')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(Xk[i, 0], cmap='gray')
    plt.title('Recovered Reflectivity')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(X[i, 0], cmap='gray')
    plt.title('Ground Truth Reflectivity')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'2D_epoch_{epoch}.png'))
    plt.close()


def save_plots(X, Xk, yn, args, epoch, train=False, algo='', dataset='marmousi'):
    plt.figure(figsize=(7, 10))
    plt.suptitle(algo + ' ' + dataset)
    Xk = Xk.detach().cpu().squeeze()
    X = X.detach().cpu().squeeze()
    X0 = yn.detach().cpu().squeeze()

    i = 1
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(X0[i], cmap='gray')
    plt.title('Observed Trace')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(Xk[i], cmap='gray')
    plt.title('Recovered Reflectivity')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(X[i], cmap='gray')
    plt.title('Ground Truth Reflectivity')
    plt.axis('off')

    file_name = 'img_' + str(epoch) + '.png' if train else dataset + '_eval.png'
    plt.savefig(os.path.join(args.save_path, file_name))
    plt.close()


