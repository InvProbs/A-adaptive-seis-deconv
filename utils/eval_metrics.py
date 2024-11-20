"""
SEISMIC TRACE RECONSTRUCTION METRICS
"""
import torch, math
import torch.nn as nn


# x, xk = torch.clone(X), torch.clone(Xk)
def gamma_N(X, Xk):
    # gamma_N = xk.T @ x / torch.sqrt((xk.T @ xk) * (x.T @ x))
    gamma_N_list = []
    for i in range(Xk.shape[3]):
        x = X[:, 0, :, i:i + 1]  # [bs, 352, 1]
        xk = Xk[:, 0, :, i:i + 1]
        xkTx = torch.bmm(xk.permute(0, 2, 1), x)
        xkTxk = torch.bmm(xk.permute(0, 2, 1), xk)
        xTx = torch.bmm(x.permute(0, 2, 1), x)
        gamma_N = xkTx / torch.sqrt(xkTxk * xTx)
        gamma_N_list.append((torch.sum(gamma_N) / len(gamma_N)).item())
    return sum(gamma_N_list) / len(gamma_N_list)


def quality(X, Xk):
    # Q_N = 10 * torch.log10(x.T @ x / (torch.norm(x - xk @ (xk * xk.T @ x) / (xk.T @ xk))) ** 2)
    # Q_N = 10 * log10(xTx/(norm(x-xk*(xkTx) / (xkTxk)))^2)
    Q_list = []
    for i in range(Xk.shape[3]):
        x = X[:, 0, :, i:i + 1]  # [bs, 352, 1]
        xk = Xk[:, 0, :, i:i + 1]
        xTx = torch.bmm(x.permute(0, 2, 1), x)
        xkTxk = torch.bmm(xk.permute(0, 2, 1), xk)
        xkTx = torch.bmm(xk.permute(0, 2, 1), x)
        norm = torch.norm(x - torch.bmm(xk, xkTx) / xkTxk, dim=1) ** 2
        Q_N = 10 * torch.log10(xTx / norm.unsqueeze(2))
        Q_list.append((torch.sum(Q_N) / len(Q_N)).item())
    return sum(Q_list) / len(Q_list)


def criteria_2D(X, Xk):
    X = X.permute(0, 1, 3, 2).reshape(X.shape[0], -1)[:, :, None]
    Xk = Xk.permute(0, 1, 3, 2).reshape(X.shape[0], -1)[:, :, None]
    xkTx = torch.bmm(Xk.permute(0, 2, 1), X)
    xkTxk = torch.bmm(Xk.permute(0, 2, 1), Xk)
    xTx = torch.bmm(X.permute(0, 2, 1), X)
    norm = torch.norm(X - torch.bmm(Xk, xkTx) / xkTxk, dim=1) ** 2
    Q_N = 10 * torch.log10(xTx / norm.unsqueeze(2))
    gamma_N = xkTx / torch.sqrt(xkTxk * xTx)
    return (sum(gamma_N) / len(gamma_N)).squeeze(), (sum(Q_N) / len(Q_N)).squeeze()


def PSNR(x, xk, i):
    criteria = nn.MSELoss()
    x = x.squeeze()
    xk = xk.squeeze()
    if i is not None:
        psnr = 20 * math.log10(1 / math.sqrt(criteria(xk, x)))
    else:
        psnr = 20 * math.log10(1 / math.sqrt(criteria(xk[i], x[i])))
    return psnr


def l1_loss(x, xk, i):
    criteria = nn.L1Loss()
    x = x.squeeze()
    xk = xk.squeeze()
    if i is not None:
        return criteria(xk, x)
    else:
        return criteria(xk[i], x[i])
