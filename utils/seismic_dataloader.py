import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.fftpack import fft, dct
from scipy import signal
import torch
import scipy.io
import scipy, mat73, os
from torch.utils.data import Dataset

# Saving data
train_trials = 10000
test_trials = 2000


def ricker(Nw, a=4.0):
    return signal.ricker(Nw, a)


def reflectivity(Ntrace, refl_length, Nlayers):
    refl_matrix = np.zeros((Ntrace, refl_length))
    for i in range(Ntrace):
        refl_per_trace = np.random.choice(refl_matrix.shape[1], Nlayers, replace=False)
        refl_coef = np.random.uniform(-0.4, 0.4, Nlayers)
        refl_matrix[i, refl_per_trace] = refl_coef
    return refl_matrix


def genTrace(Ntrace, refl_length, Nlayers, Nw, a, train):
    wavelet = ricker(Nw, a)
    refl_matrix = reflectivity(Ntrace, refl_length, Nlayers)
    trace_matrix = []
    for i in range(Ntrace):
        trace_matrix.append(np.convolve(refl_matrix[i], wavelet, mode='same'))
    trace_matrix = np.array(trace_matrix)
    data = np.column_stack((refl_matrix, trace_matrix))
    if train:
        np.savetxt('./data/deconv_training.dat', data)
    else:
        np.savetxt('./data/deconv_testing.dat', data)


def gen_dataloader2(bs):
    train_data = np.loadtxt('./data/deconv_training.dat')
    test_data = np.loadtxt('./data/deconv_testing.dat')
    print('this')
    trace_length = train_data.shape[1] // 2

    tr_refl = torch.Tensor(train_data[:, :trace_length])
    tr_trace = torch.Tensor(train_data[:, trace_length:])
    train_dataset = TensorDataset(tr_trace, tr_refl)
    train_dataloader = DataLoader(train_dataset, batch_size=bs)

    ts_refl = torch.Tensor(test_data[:, :trace_length])
    ts_trace = torch.Tensor(test_data[:, trace_length:])
    test_dataset = TensorDataset(ts_trace, ts_refl)
    test_dataloader = DataLoader(test_dataset, batch_size=bs)
    return train_dataloader, test_dataloader

    # plt.plot(data[0,:100])
    # plt.plot(data[0, 100:])


#
# def gen_datapair():
#     y = scipy.io.loadmat('./data/syn_data_generate_MirelCohen/CleanSig_MC.mat')['CleanSig_mtx'][:256, :].T
#     yn = scipy.io.loadmat('./data/syn_data_generate_MirelCohen/NoisySig_MC.mat')['NoisySig_mtx'][:256, :].T
#     refl = scipy.io.loadmat('./data/syn_data_generate_MirelCohen/syn_model_MC.mat')['syn_model_mtx'].T
#     data = np.column_stack((y, yn, refl))
#     np.random.shuffle(data)
#     np.savetxt('./data/1tps2_deconv_training.dat', data[:18000, :])
#     np.savetxt('./data/1tps2_deconv_testing.dat', data[18000:, :])
#
#
# def gen_dataloader(bs):
#     train_data = np.loadtxt('./data/1tps2_deconv_training.dat')
#     test_data = np.loadtxt('./data/1tps2_deconv_testing.dat')
#     trace_length = train_data.shape[1] // 3
#
#     y_tr = torch.Tensor(train_data[:, :trace_length])
#     yn_tr = torch.Tensor(train_data[:, trace_length:2 * trace_length])
#     r_tr = torch.Tensor(train_data[:, 2 * trace_length:3 * trace_length])
#     train_dataset = TensorDataset(y_tr, yn_tr, r_tr)
#     train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
#
#     y_ts = torch.Tensor(test_data[:, :trace_length])
#     yn_ts = torch.Tensor(test_data[:, trace_length:2 * trace_length])
#     r_ts = torch.Tensor(test_data[:, 2 * trace_length: 3 * trace_length])
#     test_dataset = TensorDataset(y_ts, yn_ts, r_ts)
#     test_dataloader = DataLoader(test_dataset, batch_size=bs)
#     return train_dataloader, test_dataloader, len(train_data), len(test_data)
#
#
# def readFile():
#     true_wavelet = scipy.io.loadmat('./data/1tps_TrueWavelet.mat')['TrueWavelet']
#     W = scipy.io.loadmat('./data/1tps_W_mtx.mat')['W']
#     return true_wavelet, W


def gen_datapair():
    # y = scipy.io.loadmat('../data/reflectivity_train_1D.mat')['quantized_trace'].T
    # refl = scipy.io.loadmat('../data/reflectivity_train_2.mat')['Ref'].T
    y = mat73.loadmat('../data/reflectivity_train_1D_noisy.mat')['quantized_trace']  # (352, 5000)
    refl = mat73.loadmat('../data/reflectivity_train_1D_noisy.mat')['Ref']  # (352, 5000)
    data = np.column_stack((y, refl))
    np.random.shuffle(data)
    np.savetxt('../data/train_1D_noisy.dat', data)

    # y = scipy.io.loadmat('../data/reflectivity_test_2.mat')['quantized_trace'].T
    # refl = scipy.io.loadmat('../data/reflectivity_test_2.mat')['Ref'].T
    y = mat73.loadmat('../data/reflectivity_test_1D_noisy.mat')['quantized_trace']  # (352, 1000)
    refl = mat73.loadmat('../data/reflectivity_test_1D_noisy.mat')['Ref']  # (352, 1000)
    data = np.column_stack((y, refl))
    np.savetxt('../data/test_1D_noisy.dat', data)


def gen_dataloader(bs):
    train_data = np.loadtxt('../data/train_1D_noisy.dat')
    test_data = np.loadtxt('../data/test_1D_noisy.dat')
    # train_data = np.loadtxt('../data/naveed_training.dat')
    # test_data = np.loadtxt('../data/naveed_testing.dat')[:1000, :]
    trace_length = train_data.shape[1] // 2

    y_tr = torch.Tensor(train_data[:, :trace_length])
    r_tr = torch.Tensor(train_data[:, trace_length:])
    train_dataset = TensorDataset(y_tr, r_tr)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False)

    y_ts = torch.Tensor(test_data[:, :trace_length])
    r_ts = torch.Tensor(test_data[:, trace_length:])
    test_dataset = TensorDataset(y_ts, r_ts)
    test_dataloader = DataLoader(test_dataset, batch_size=bs)
    return train_dataloader, test_dataloader, len(train_data), len(test_data)


def readFile():
    true_wavelet = mat73.loadmat('../data/reflectivity_train_1D.mat')['W']
    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), 352)
    return true_wavelet, W[25:377, :]


class Custom1DDataset(Dataset):
    def __init__(self, data_dir, refl_transform=None, trace_transform=None):
        self.root_dir = data_dir
        self.refl_transform = refl_transform
        self.trace_transform = trace_transform
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        ntraces = 352
        trace_length = 352
        refl = scipy.io.loadmat(path)['syn_model']
        trace = scipy.io.loadmat(path)['NoisySig']
        # refl = scipy.io.loadmat(path)['refl_sample'].reshape(ntraces, trace_length).T
        # trace = scipy.io.loadmat(path)['trace_sample'].reshape(ntraces, trace_length).T
        if self.refl_transform:
            refl = self.refl_transform(refl)
        if self.trace_transform:
            trace = self.trace_transform(trace)
        return trace, refl


def gen_dataloader1D(args):
    # train_path = '../data/2D_352traces_training/' #'../data/syn_data_generate_MirelCohen/seis2D_train'
    # test_path = '../data/2D_352traces_testing/' # '../data/syn_data_generate_MirelCohen/seis2D_test'
    train_path = '../data/syn_data_generate_MirelCohen/seis1D_train'
    test_path = '../data/syn_data_generate_MirelCohen/seis1D_test'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Custom1DDataset(train_path, refl_transform=transform, trace_transform=transform)
    test_dataset = Custom1DDataset(test_path, refl_transform=transform, trace_transform=transform)
    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
    return tr_loader, len(train_dataset), ts_loader, len(test_dataset)


def gen_datapair2D():
    y = mat73.loadmat('../data/reflectivity_train_2D_352traces.mat')['quantized_trace']
    refl = mat73.loadmat('../data/reflectivity_train_2D_352traces.mat')['Ref']
    data = np.column_stack((y, refl))
    np.random.shuffle(data)  # ???
    np.savetxt('../data/training_2D_352traces.dat', data)

    y = mat73.loadmat('../data/reflectivity_test_2D_352traces.mat')['quantized_trace']
    refl = mat73.loadmat('../data/reflectivity_test_2D_352traces.mat')['Ref']
    data = np.column_stack((y, refl))
    np.savetxt('../data/testing_2D_352traces.dat', data)

#
# def gen_dataloader2D(args):
#     if args.train:
#         ntraces = 352  # 352 or 5 or 50
#         data = np.loadtxt('../data/training_2D_352traces.dat')
#         tot_length = data.shape[1] // 2
#         trace_length = tot_length // ntraces
#         y_tr = torch.Tensor(data[:, :tot_length]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         r_tr = torch.Tensor(data[:, tot_length:]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         train_dataset = TensorDataset(y_tr, r_tr)
#         train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         train_length = len(data)
#
#         data = np.loadtxt('../data/testing_2D_352traces.dat')  # [:1000, :]
#         y_ts = torch.Tensor(data[:, :tot_length]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         r_ts = torch.Tensor(data[:, tot_length:]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         test_dataset = TensorDataset(y_ts, r_ts)
#         test_dataloader = DataLoader(test_dataset, batch_size=32)
#         test_length = len(data)
#         return train_dataloader, test_dataloader, train_length, test_length
#     else:
#         ntraces = 352  # 352 or 5 or 50
#         test_data = np.loadtxt('../data/testing_2D_352traces.dat')  # [:1000, :]
#         tot_length = test_data.shape[1] // 2
#         trace_length = tot_length // ntraces
#
#         y_ts = torch.Tensor(test_data[:, :tot_length]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         r_ts = torch.Tensor(test_data[:, tot_length:]).reshape(-1, 1, ntraces, trace_length).permute(0, 1, 3, 2)
#         test_dataset = TensorDataset(y_ts, r_ts)
#         test_dataloader = DataLoader(test_dataset, batch_size=32)
#         return test_dataloader, len(test_data)


class Custom2DDataset(Dataset):
    def __init__(self, data_dir, refl_transform=None, trace_transform=None):
        self.root_dir = data_dir
        self.refl_transform = refl_transform
        self.trace_transform = trace_transform
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        ntraces = 352
        trace_length = 352
        # refl = scipy.io.loadmat(path)['syn_model']
        # trace = scipy.io.loadmat(path)['NoisySig']
        refl = scipy.io.loadmat(path)['refl_sample'].reshape(ntraces, trace_length).T
        trace = scipy.io.loadmat(path)['trace_sample'].reshape(ntraces, trace_length).T
        if self.refl_transform:
            refl = self.refl_transform(refl)
        if self.trace_transform:
            trace = self.trace_transform(trace)
        return trace, refl


def gen_dataloader2D(args):
    train_path = '../data/seis_deconv/2D_352traces_training/' #'../data/syn_data_generate_MirelCohen/seis2D_train'
    test_path = '../data/seis_deconv/2D_352traces_testing' # '../data/syn_data_generate_MirelCohen/seis2D_test'
    # train_path = '../data/syn_data_generate_MirelCohen/2D_train_SNR30'
    # test_path = '../data/syn_data_generate_MirelCohen/2D_test_SNR30'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Custom2DDataset(train_path, refl_transform=transform, trace_transform=transform)
    test_dataset = Custom2DDataset(test_path, refl_transform=transform, trace_transform=transform)
    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
    return tr_loader, len(train_dataset), ts_loader, len(test_dataset)

# plt.plot(r_tr[0,0,0,:])
# plt.plot(y_tr[0,0,0,:])
# plt.figure(); plt.imshow(r_tr[0,0], cmap='gray')
# plt.figure(); plt.imshow(y_tr[0,0], cmap='gray')


def readFile2D():
    true_wavelet = mat73.loadmat('../data/seis_deconv/reflectivity_test_1D.mat')['W']
    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), 352)
    return true_wavelet, W[25:377, :]


def gen_mtxW(points, a=4):
    vec = scipy.signal.ricker(points, a)
    vec = vec / np.max(vec)
    W = scipy.linalg.convolution_matrix(vec.squeeze(), 352)
    return vec, W[25:377, :]


def read_W_real():
    W = scipy.io.loadmat('../data/real_3/150_31.mat')['W_shift']
    W /= np.max(W)
    W = scipy.linalg.convolution_matrix(W.squeeze(), 352)
    cut = (W.shape[0] - 352) // 2
    W = torch.Tensor(W[cut:W.shape[0] - cut, :])
    return W


def image2patch(trace, H, W):
    # trace in shape 2800x13601
    # patch in shape  352x50
    trace_H, trace_W = trace.shape
    nH, nW = trace_H // H, trace_W // W
    patches = np.zeros((nH * nW, 1, H, W))
    for i in range(nH):
        for j in range(nW):
            patch = trace[i * H:(i + 1) * H, j * W:(j + 1) * W]
            patches[i * nW + j, 0] = patch
    return patches


def patch2image(patches):
    recovered_trace = np.zeros((8 * 272, 13600))
    for i in range(8):
        for j in range(272):
            patch = patches[i * 8 + j, 0]
            recovered_trace[i * 352:(i + 1) * 352, j * 50:(j + 1) * 50] = patch
    return recovered_trace


def readMarmousi2_conv_2D(ds, args):
    if ds == 1:
        trace = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['W']
    if ds == 2:
        trace = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['W']
    if ds == 3:
        trace = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['W']

    H, W = trace.shape
    ntraces = 352
    trace, reflectivity = trace[:(H // 352) * 352, :(W // ntraces) * ntraces], reflectivity[:(H // 352) * 352, :(W // ntraces) * ntraces]
    trace_patch, reflectivity_patch = torch.Tensor(image2patch(trace, 352, ntraces)), torch.Tensor(image2patch(reflectivity, 352, ntraces))
    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), 352)

    marmousi2_dataset = TensorDataset(trace_patch, reflectivity_patch)
    marmousi2_dataloader = DataLoader(marmousi2_dataset, batch_size=args.batch_size_val)
    return marmousi2_dataloader, len(marmousi2_dataset), W[25:377, :]


def readMarmousi2_conv_1D(ds):
    if ds == 1:
        trace = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_full.mat')['W']
    if ds == 2:
        trace = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by2.mat')['W']
    if ds == 3:
        trace = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['trace']
        reflectivity = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['reflectivity']
        true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['W']

    W, H = trace.shape
    pad = 0
    trace_effL = 352 - 2 * pad
    trace, reflectivity = trace[:(W // trace_effL) * trace_effL, :], reflectivity[:(W // trace_effL) * trace_effL, :]
    trace, reflectivity = torch.Tensor(image2patch(trace, 352, 1)), torch.Tensor(image2patch(reflectivity, 352, 1))
    # trace = np.pad(trace, ((0, 0), (pad, pad)), mode='constant', constant_values=0)
    # reflectivity = np.pad(reflectivity, ((0, 0), (pad, pad)), mode='constant', constant_values=0)

    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), trace_effL)

    trace, reflectivity = torch.Tensor(trace), torch.Tensor(reflectivity)

    marmousi2_dataset = TensorDataset(trace, reflectivity)
    marmousi2_dataloader = DataLoader(marmousi2_dataset, batch_size=32)
    return marmousi2_dataloader, len(marmousi2_dataset), W[25:377, :]


# def readRealData_conv_2D():
#     # trace = scipy.io.loadmat('../data/PG_real_data/gm99_7stk.mat')['Traces']
#     # trace = scipy.io.loadmat('../data/PG_real_data/31_81_pr_dec.mat')['Data']
#     # trace = scipy.io.loadmat('../data/csg2.mat')['d']
#     # true_wavelet = scipy.io.loadmat('../data/real_2/150_31.mat')['W_shift']
#     # trace = scipy.io.loadmat('../data/real_2/real.mat')['data']
#     # true_wavelet = scipy.io.loadmat('../data/real_3/150_31.mat')['W_shift']
#
#     trace = scipy.io.loadmat('../data/real_3/real.mat')['data']
#     H, W = trace.shape
#     # cmp = np.zeros((18, H, 33))
#     # for i in range(18):
#     #     cmp[i] = trace[:, i*33:(i+1)*33]
#     # cmp = np.sum(cmp, axis=0)/18
#     # pad = (352 - 33)//2
#     # trace = np.pad(cmp, ((0, 0), (pad, pad+1)), mode='constant', constant_values=0)
#     # trace = trace[50:, :]
#     # trace = trace[::3, :]
#
#     ntraces = 352
#     trace = trace[:(H // 352) * 352, :(W // ntraces) * ntraces]
#     trace_patch = torch.Tensor(image2patch(trace, 352, ntraces))
#     # W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), 352)
#
#     real_dataset = TensorDataset(trace_patch, trace_patch)
#     real_dataloader = DataLoader(real_dataset, batch_size=16)
#     return real_dataloader, len(real_dataset)#, W[25:377, :]

def readRealData_conv_2D():
    trace = scipy.io.loadmat('../data/Book_Data/Book_Seismic_Data_gain_bpf_sdecon_gain_sorted_nmo_corrected_stacked.mat')['Dstacked']
    # trace = scipy.io.loadmat('../data/trace_afterAGC.mat')['Dg']

    trace[np.isnan(trace)] = 0

    pad = (352 - trace.shape[1]) // 2
    trace = np.pad(trace, ((25, 33), (pad, pad + 1)), mode='constant', constant_values=0)  # [1527, 352]
    H, W = trace.shape
    ntraces = 352
    trace = trace[:(H // (352-50) * (352-50) + 50), :]
    trace_patch = np.zeros((H // (352-50), 1, 352, W))
    for i in range(len(trace_patch)):
        start = i * (351 - 50)
        trace_patch[i, 0] = trace[start:(start + 352), :]
    trace_patch = torch.Tensor(trace_patch)
    real_dataset = TensorDataset(trace_patch, trace_patch)
    real_dataloader = DataLoader(real_dataset, batch_size=1)
    return real_dataloader, len(real_dataset) #, W[25:377, :]


    # pad = 143
    # W = 65
    # plt.figure(figsize=(2, 7))
    # plt.imshow(trace[:,pad:pad + W], interpolation='nearest', aspect='auto')



""" CURRENTLY WORKING VERSION """
# def readRealData_conv_2D():
#     trace = \
#     scipy.io.loadmat('../data/Book_Data/Book_Seismic_Data_gain_bpf_sdecon_gain_sorted_nmo_corrected_stacked.mat')[
#         'Dstacked']
#     trace[np.isnan(trace)] = 0
#     H, W = trace.shape
#     pad = (352 - trace.shape[1]) // 2
#     trace = np.pad(trace, ((25, 0), (pad, pad + 1)), mode='constant', constant_values=0)  # [1502, 352]
#
#     ntraces = 352
#     trace = trace[:(H // 352) * 352, :]
#     trace_patch = torch.Tensor(image2patch(trace, 352, ntraces))
#
#     real_dataset = TensorDataset(trace_patch, trace_patch)
#     real_dataloader = DataLoader(real_dataset, batch_size=16)
#     return real_dataloader, len(real_dataset)  # , W[25:377, :]


# def readRealData_conv_1D():
#     trace = scipy.io.loadmat('../data/PG_real_data/gm99_7stk.mat')['Traces']
#     # trace = scipy.io.loadmat('../data/PG_real_data/31_81_pr_dec.mat')['Data']
#     true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['W']
#
#     trace = trace[2000:, :]
#     H, W = trace.shape
#     pad = 0
#     trace_effL = 352 - 2 * pad
#     trace= trace[:(W // trace_effL) * trace_effL, :]
#     trace = torch.Tensor(image2patch(trace, 352, 1))
#     W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), trace_effL)
#
#     trace = torch.Tensor(trace)
#
#     real_dataset = TensorDataset(trace, trace)
#     real_dataloader = DataLoader(real_dataset, batch_size=32)
#     return real_dataloader, len(real_dataset), W[25:377, :]


def readRealData_conv_1D():
    # trace = scipy.io.loadmat('../data/csg2.mat')['d']
    # trace = scipy.io.loadmat('../data/Book_Data/Book_Seismic_Data_gain_z_bpf.mat')['Dbpf']
    trace = scipy.io.loadmat('../data/Book_Data/Book_Seismic_Data_gain_bpf_sdecon_gain_sorted_nmo_corrected_stacked.mat')[
        'Dstacked']
    trace[np.isnan(trace)] = 0
    true_wavelet = scipy.io.loadmat('../data/marmoussi_conv_ds_by3.mat')['W']
    # trace = trace[::2, :]
    # trace = trace[500:, :]

    H, W = trace.shape
    pad = 0
    trace_effL = 352 - 2 * pad
    trace = trace[:(H // trace_effL) * trace_effL, :]
    trace = torch.Tensor(image2patch(trace, 352, 1))
    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), trace_effL)
    trace = torch.Tensor(trace)
    real_dataset = TensorDataset(trace, trace)
    real_dataloader = DataLoader(real_dataset, batch_size=32)
    return real_dataloader, len(real_dataset), W[25:377, :]
