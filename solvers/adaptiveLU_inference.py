import matplotlib, sys, configargparse
sys.path.append("../")
from utils.seismic_dataloader import *
from operators import seismic_operators as op
from utils.save_plots import *
from networks import LU_net as nets
from utils.misc import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.eval_metrics import *
from pandas import *

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--eta', type=float, default=0.01, help='initial eta')
parser.add_argument("--location", type=str, default="work", help="home or work directory")
parser.add_argument("--file_name", type=str, default="adaptiveLU/", help="saving directory")
parser.add_argument("--save_path", type=str, default="", help="saving directory")
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")

parser.add_argument('--lr', type=float, default=1e-4, help='training lr')
parser.add_argument('--maxiters', type=int, default=8, help='Main max iterations')
parser.add_argument('--n_epochs', default=200)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=16, help='val and test batch size')
parser.add_argument('--pretrain', type=bool, default=True, help='if load utils and resume training')
parser.add_argument('--train', type=bool, default=False, help='training or eval mode')
parser.add_argument('--A_adaptive_LU_mode', type=bool, default=False, help='eval with regular LU or adaptive LU')

parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
parser.add_argument('--noise_level', type=float, default=0.01)
parser.add_argument('--nc', type=int, default=1, help='number of channels in an image')

parser.add_argument('--mu', type=float, default=10, help='coefficient of ||x-z||')
parser.add_argument('--lamb', type=float, default=0.1, help='coefficient of ||x-z||')
parser.add_argument('--tau', type=float, default=0.01, help='coefficient of ||x-z||')

args = parser.parse_args()
args.shared_eta = True
print(args)
""" set seed and device """
set_seed(SEED=110)

""" Load Data """
train_loader, tr_length, test_loader, ts_length = gen_dataloader2D(args)
true_wavelet, W = readFile2D()
W = torch.Tensor(W).to(args.device)
create_save_path(args)

""" Network Setup """
# estimated forward model W0
_, _, dataloader, ts_length = gen_dataloader2D(args)
_, W_file = readFile2D()
W0 = scipy.signal.ricker(51, 14)
W0 /= np.max(W0)
W0 = scipy.linalg.convolution_matrix(W0.squeeze(), 352)
W0 = torch.Tensor(W0[25:377, :]).to(args.device)

# true forward model W
decay_factor = 0.001
decay_vec = torch.Tensor(np.exp(- decay_factor * np.arange(352))).unsqueeze(1).to(args.device)
W = torch.Tensor(W_file).to(args.device) + 0.01 * torch.randn_like(W0).to(args.device)
true_forward_op = op.convolution(torch.clone(W), trace_length=352).to(args.device)
measurement_process = op.OperatorPlusNoise(true_forward_op, noise_sigma=args.noise_level * 0.01).to(args.device)

# setup network and training parameters
netA = op.adaptive_convolution(torch.clone(W0), trace_length=352).to(args.device)
optA = torch.optim.Adam(netA.parameters(), lr=1e-4)
netR = nets.single_layer_conv2D_y(args).to(args.device)
est_forward_op = op.convolution(W0, trace_length=352).to(args.device)
invBlock = nets.inverse_block_prox2D_y(est_forward_op, netR, args).to(args.device)

unet = nets.UNet().to(args.device)
unet.load_state_dict(torch.load('../saved_models/unet_sigma=0.05.state')['state_dict'])

DnCNN = nets.single_layer_conv2D_y(args).to(args.device)
DnCNN.load_state_dict(torch.load('../saved_models/new_DnCNN/eps=0.01_0609-1114/epoch_99.state')['state_dict'])

""" Begin Eval ..."""
criteria = nn.MSELoss()
args.load_path = '../saved_models/LU_sigma=0.1.state'
invBlock.load_state_dict(torch.load(args.load_path)['state_dict'])
eta = invBlock.sigmoid(invBlock.eta)
netR.eval()

loss_meters = [AverageMeter() for _ in range(1)]
mse_meter = AverageMeter()
gamma_meter = AverageMeter()
quality_meter = AverageMeter()
with tqdm(total=(ts_length - ts_length % args.batch_size)) as _tqdm:
    _tqdm.set_description('epoch: {}/{}'.format(1, 1))
    for _, X in dataloader:
        bs = X.shape[0]
        X = X.to(args.device).type(torch.cuda.FloatTensor)
        yn = measurement_process(X).detach()
        maxVal, yn, X = normalize(yn, X, bs)
        X0, Xk, Zk = torch.clone(yn), torch.clone(yn), torch.clone(yn)
        Xk = DnCNN(yn, X0)
        X0 = torch.clone(Xk)
        Zk = torch.clone(Xk)

        if not args.A_adaptive_LU_mode:
            for k in range(args.maxiters):
                Xk = invBlock(Xk, yn, True)
            plot_reflectivity(X, yn, Xk, args, -3)
        else:
            for k in range(args.maxiters):
                # update Z
                netA.eval()
                scale_coef = 1 / torch.max(torch.abs(netA.A))
                Zk = torch.inverse(1 / scale_coef ** 2 * netA.A.T  @ netA.A + args.mu * torch.eye(352).to(args.device)) @ (args.mu * Xk + netA.A.T @ yn / scale_coef)
                Zk = Zk.detach()

                # update A
                Zk.requires_grad_(False)
                netA.train()
                loss_list = []
                for i in range(500):
                    yk = netA(Zk)
                    optA.zero_grad()
                    lossA = criteria(yn, yk) + args.tau * torch.norm(netA.A)
                    lossA.backward()
                    optA.step()
                    loss_list.append(lossA.item())

                # update X
                netA.eval()
                Xk = invBlock.R(yn, Xk - eta * (Xk - Zk))

        mse = criteria(Xk.squeeze(), X.squeeze())
        gamma = gamma_N(X, Xk)
        qual = quality(X, Xk)
        mse_meter.update(mse, bs)
        gamma_meter.update(gamma, bs)
        quality_meter.update(qual, bs)
        _tqdm.set_postfix({'mse': f'{mse_meter.avg:.6f}', 'gamma': f'{gamma_meter.avg:.6f}',
                           'qual': f'{quality_meter.avg:.6f}'})
        _tqdm.update(bs)
    plot_reflectivity(X, yn, Xk, args, -1)

