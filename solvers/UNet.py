""" Train and eval of UNet in seismic deconvolution """

import matplotlib, sys
sys.path.append("../")
from utils.seismic_dataloader import *
from operators import seismic_operators as op
from utils.save_plots import *
from networks import LU_net as nets
import torch.optim as optim
from utils.misc import *
import configargparse
import os
from tqdm import tqdm
from utils.eval_metrics import *

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--n_epochs', default=400)
parser.add_argument("--location", type=str, default="work", help="home or work directory")
parser.add_argument('--lr', type=float, default=1e-4, help='training lr')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=16, help='val and test batch size')
parser.add_argument("--file_name", type=str, default="Unet/", help="saving directory")
parser.add_argument("--save_path", type=str, default="", help="saving directory")
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='./saved_models/')
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
parser.add_argument('--noise_level', type=float, default=0.01)
parser.add_argument('--pretrain', type=bool, default=True, help='if load utils and resume training')
parser.add_argument('--train', type=bool, default=False, help='training or eval mode')
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--sched_step', type=int, default=100)
args = parser.parse_args()
print(args)
""" set seed and device """
set_seed(SEED=110)

""" Load Data """
train_loader, tr_length, test_loader, ts_length = gen_dataloader2D(args)
true_wavelet, W = readFile2D()
W = torch.Tensor(W).to(args.device)
create_save_path(args)

""" Network Setup """
forward_operator = op.convolution(W, trace_length=352).to(args.device)
measurement_process = op.OperatorPlusNoise(forward_operator, noise_sigma=args.noise_level).to(args.device)
invBlock = nets.UNet().to(args.device)
print("# Parmeters: ", sum(a.numel() for a in invBlock.parameters()))

""" Begin Training ..."""
invBlock.train()
opt = torch.optim.AdamW(invBlock.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer=opt, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
criteria = nn.MSELoss()
if args.pretrain:
    load_model(args, invBlock, '')

if args.train:
    for epoch in range(args.n_epochs):
        epoch_losses = AverageMeter()
        ts_losses = AverageMeter()
        with tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for yn, X in train_loader:
                # data preprocessing
                bs = X.shape[0]
                yn = yn.to(args.device).type(torch.cuda.FloatTensor)
                yn += torch.randn_like(yn) * args.noise_level
                X = X.to(args.device).type(torch.cuda.FloatTensor)
                maxVal, yn, X = normalize(yn, X, bs)
                X0 = torch.clone(yn)

                # reconstruction prediction
                Xk = invBlock(X0)

                # training
                loss = criteria(Xk.squeeze(), X.squeeze())
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_losses.update(loss.item(), bs)
                torch.cuda.empty_cache()
                _tqdm.set_postfix({'tr': f'{epoch_losses.avg:.6f}', 'ts': f'{ts_losses.avg:.6f}'})
                _tqdm.update(bs)

            # save model and visualization
            if (epoch + 1) % 25 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': invBlock.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))
            if (epoch + 1) % 25 == 0:
                save_plots(X, Xk, yn, args, -1, train=True, algo='UNet', dataset='Synthetic')

            """ Validation """
            with torch.no_grad():
                for yn, X in test_loader:
                    bs = X.shape[0]
                    yn = yn.to(args.device).type(torch.cuda.FloatTensor)
                    X = X.to(args.device).type(torch.cuda.FloatTensor)
                    maxVal, yn, X = normalize(yn, X, bs)
                    X0 = torch.clone(yn)
                    Xk = invBlock(X0)
                    loss = criteria(Xk.squeeze(), X.squeeze())
                    ts_losses.update(loss.item(), bs)
                    _tqdm.set_postfix({'tr': f'{epoch_losses.avg:.6f}', 'ts': f'{ts_losses.avg:.6f}'})
                    _tqdm.update(bs)

else:
    print('Begin evaluation...')

    """ Evaluation on in-distribution synthetic dataset """
    loss_meters = [AverageMeter() for _ in range(1)]
    mse_meter = AverageMeter()
    gamma_meter = AverageMeter()
    quality_meter = AverageMeter()
    with tqdm(total=(ts_length - ts_length % args.batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, args.n_epochs))
        with torch.no_grad():
            for yn, X in test_loader:
                bs = X.shape[0]
                X = X.to(args.device).type(torch.cuda.FloatTensor)
                yn = measurement_process(X)
                maxVal, yn, X = normalize(yn, X, bs)
                X0 = torch.clone(yn)
                Xk = invBlock(X0)

                mse = criteria(Xk.squeeze(), X.squeeze())
                gamma = gamma_N(X, Xk)
                qual = quality(X, Xk)
                mse_meter.update(mse, bs)
                gamma_meter.update(gamma, bs)
                quality_meter.update(qual, bs)
                _tqdm.set_postfix({'mse': f'{mse_meter.avg:.6f}', 'gamma': f'{gamma_meter.avg:.6f}',
                                   'qual': f'{quality_meter.avg:.6f}'})
                _tqdm.update(bs)
            save_plots(X, Xk, yn, args, -1, train=False, algo='UNet', dataset='Synthetic Data')



    """ Evaluation on Marmousi2 dataset """
    # load marmousi2 data and initial forward model W0
    marmousi2_loader, marmousi2_length, W0 = readMarmousi2_conv_2D(2, args)
    W0 = torch.Tensor(W0).to(args.device)

    # add decay factor to obtain the true forward model W
    decay_factor = 0.01
    decay_vec = torch.Tensor(np.exp(- decay_factor * np.arange(352))).unsqueeze(1).to(args.device)
    W = W0 * decay_vec.repeat(1, 352)
    forward_operator = op.convolution(W, trace_length=352).to(args.device)
    measurement_process = op.OperatorPlusNoise(forward_operator, noise_sigma=args.noise_level * 0.1).to(args.device)

    # begin evaluation
    loss_meters = [AverageMeter() for _ in range(1)]
    mse_meter = AverageMeter()
    gamma_meter = AverageMeter()
    quality_meter = AverageMeter()
    with tqdm(total=(marmousi2_length - marmousi2_length % args.batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, args.n_epochs))
        with torch.no_grad():
            for yn, X in marmousi2_loader:
                bs = X.shape[0]
                X = X.to(args.device).type(torch.cuda.FloatTensor)
                yn = measurement_process(X)
                maxVal, yn, X = normalize(yn, X, bs)
                X0 = torch.clone(yn)
                Xk = invBlock(X0)

                X = X[:, :, 25:(352 - 25), :]
                Xk = Xk[:, :, 25:(352 - 25), :]
                yn = yn[:, :, 25:(352 - 25), :]
                mse = criteria(Xk.squeeze(), X.squeeze())
                gamma, qual = criteria_2D(X, Xk)
                mse_meter.update(mse, bs)
                gamma_meter.update(gamma, bs)
                quality_meter.update(qual, bs)
                _tqdm.set_postfix({'mse': f'{mse_meter.avg:.6f}', 'gamma': f'{gamma_meter.avg:.6f}',
                                   'qual': f'{quality_meter.avg:.6f}'})
                _tqdm.update(bs)
            save_plots(X, Xk, yn, args, -1, train=False, algo='UNet', dataset='marmousi')
