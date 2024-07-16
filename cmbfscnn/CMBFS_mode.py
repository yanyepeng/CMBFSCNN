
# -*- coding: utf-8 -*-



import torch
from . import utils
from . import CNN_models as Cm
from . import generate_data as gd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt


class Foreground_subtraction_model(gd.Data_preprocessing):
    def __init__(self, data_dir='DATA/', freqs = np.array([]), output_freq = 220, component ="Q", full_sky_map = False,
                 is_half_split_map=True, using_ilc_cmbmap=True, result_dir = 'DATA_results/'):

        if not data_dir is None:
            self._save_dir = data_dir
            self._cread_file_name
        super(gd.Data_preprocessing, self).__init__()
        self.result_dir = result_dir
        self.freqs = freqs
        self.output_freq = output_freq
        self.component = component
        self.full_sky_map = full_sky_map
        self.is_half_split_map = is_half_split_map
        self.using_ilc_cmbmap = using_ilc_cmbmap
        self._creat_file_n
        self.l1_loss = torch.nn.L1Loss()

    @property
    def output_freq_index(self):
        return np.where(self.freqs==self.output_freq)[0][0]


    @property
    def _creat_file_n(self):
        dir_key = {}
        dir_key['records_Tmap_dir'] = self.result_dir + 'training_records_T_comp/'
        dir_key['records_Qmap_dir'] = self.result_dir+'training_records_Q_comp/'
        dir_key['records_Umap_dir'] = self.result_dir + 'training_records_U_comp/'
        dir_key['output_Tmap_dir'] = self.result_dir + 'output_T_comp/'
        dir_key['output_Qmap_dir'] = self.result_dir + 'output_Q_comp/'
        dir_key['output_Umap_dir'] = self.result_dir + 'output_U_comp/'

        for key, value in dir_key.items():
            setattr(self, key, value)



    def LrDecay(self, iteration, ith_iteration, lr=0.1, lr_min=1e-7):
        gamma = (lr_min / lr) ** (1. / iteration)
        return lr * gamma ** ith_iteration


    def loss_fft(self, cmb_false, cmb_true, lambda_l=0.5):
        cmb_false_fft = torch.abs(torch.fft.fftn(cmb_false, dim=(2, 3))) / cmb_true.shape[3]
        cmb_true_fft = torch.abs(torch.fft.fftn(cmb_true, dim=(2, 3))) / cmb_true.shape[3]
        return lambda_l * self.l1_loss(cmb_true_fft, cmb_false_fft)

    def loss_function(self, output, target, using_loss_fft = True):
        if using_loss_fft:
            return self.l1_loss(output, target) + self.loss_fft(output, target, lambda_l = 1)
        else:
            return self.l1_loss(output, target)



    def cnn_m(self, model_name):
        if model_name == 'CMBFSCNN_level3':
            self._net = Cm.CMBFSCNN_level3(in_channels = len(self.freqs), out_channels = 1, n_feats = 16)
        elif model_name == 'CMBFSCNN_level4':
            self._net = Cm.CMBFSCNN_level4(in_channels = len(self.freqs), out_channels = 1, n_feats = 16)
        elif model_name == 'UNet':
            self._net = Cm.UNet(in_channels = len(self.freqs), out_channels = 1)
        else:
            print("Error: Please set model_name correctly. model_name = 'CMBFSCNN_level3' or 'CMBFSCNN_level4' or UNet ")

    def MAD(self, out, target):
        return np.mean(abs(out-target))

    def _load_batch_data(self, index_sample, data_tyep = 'training_set', batch_size=None, half_sp = 1, comp=None):
        '''

        :param index_sample: map index in data_set
        :param data_tyep: dataset_type = 'traing_set' or 'validation_set' or 'testing_set'
        :return:
        '''
        if comp is None:
            comp = self.component
        if batch_size is None:
            batch_size = self.batch_size

        if batch_size >len(index_sample):
            batch_index = np.random.choice(index_sample, batch_size, replace=True)  # Note: replace=False
        else:
            batch_index = np.random.choice(index_sample, batch_size, replace=False)
        if data_tyep == 'training_set':
            if self.full_sky_map:
                if self.using_ilc_cmbmap:
                    cmb_dir = getattr(self, 'dir_cmb_ilc_flat_{}'.format(data_tyep[0:3]))
                else:
                    cmb_dir = getattr(self, 'dir_cmb_obs_flat_{}'.format(data_tyep[0:3]))
                total_dir = getattr(self, 'dir_total_obs_flat_{}'.format(data_tyep[0:3]))
            else:
                if self.using_ilc_cmbmap:
                    cmb_dir = getattr(self, 'dir_cmb_ilc_flat_block_{}'.format(data_tyep[0:3]))
                else:
                    cmb_dir = getattr(self, 'dir_cmb_obs_flat_block_{}'.format(data_tyep[0:3]))
                total_dir = getattr(self, 'dir_total_obs_flat_block_{}'.format(data_tyep[0:3]))
            load = gd.Load_data(map_nums=batch_index, component=comp,
                                               input_dir=total_dir, target_dir=cmb_dir,
                                               output_freq=self.output_freq_index)
            cmb_batch, total_batch = load.data()
        else:
            if self.full_sky_map:
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_dir = getattr(self, 'dir_cmb_ilc_flat_{}_{}'.format(data_tyep[0:3],half_sp))
                    else:
                        cmb_dir = getattr(self, 'dir_cmb_obs_flat_{}_{}'.format(data_tyep[0:3], half_sp))
                    total_dir = getattr(self, 'dir_total_obs_flat_{}_{}'.format(data_tyep[0:3],half_sp))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_dir = getattr(self, 'dir_cmb_ilc_flat_{}'.format(data_tyep[0:3]))
                    else:
                        cmb_dir = getattr(self, 'dir_cmb_obs_flat_{}'.format(data_tyep[0:3]))
                    total_dir = getattr(self, 'dir_total_obs_flat_{}'.format(data_tyep[0:3]))
            else:
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_dir = getattr(self, 'dir_cmb_ilc_flat_block_{}_{}'.format(data_tyep[0:3], half_sp))
                    else:
                        cmb_dir = getattr(self, 'dir_cmb_obs_flat_block_{}_{}'.format(data_tyep[0:3], half_sp))
                    total_dir = getattr(self, 'dir_total_obs_flat_block_{}_{}'.format(data_tyep[0:3], half_sp))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_dir = getattr(self, 'dir_cmb_ilc_flat_block_{}'.format(data_tyep[0:3]))
                    else:
                        cmb_dir = getattr(self, 'dir_cmb_obs_flat_block_{}'.format(data_tyep[0:3]))
                    total_dir = getattr(self, 'dir_total_obs_flat_block_{}'.format(data_tyep[0:3]))
            load = gd.Load_data(map_nums=batch_index, component=comp,
                                               input_dir=total_dir, target_dir=cmb_dir,
                                               output_freq=self.output_freq_index)
            cmb_batch, total_batch = load.data()

        return cmb_batch, total_batch


    def net_train(self, batch_size=12, learning_rate=0.01, num_train=1000, num_validation = 200, device_ids = [0,1], iteration=3e4,
                  CNN_model='CMBFSCNN_level3', using_loss_fft=True, repeat_n=3):
        self.records_dir = getattr(self, 'records_{}map_dir'.format(self.component))
        self._creat_file(self.records_dir)
        iteration = int(iteration)
        self.batch_size = batch_size
        if iteration<100:
            ns = 2
        elif iteration<1000:
            ns = 10
        elif iteration<10000:
            ns = 100
        elif iteration>10000:
            ns = 500
        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        self.cnn_m(CNN_model)
        self._net = nn.DataParallel(self._net.to(device), device_ids=device_ids)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=learning_rate, weight_decay=0)
        pbar = tqdm(total=iteration, miniters=2)
        pbar.set_description('Training CNN model for {} map'.format(self.component))
        train_loss, valid_loss, train_MAD, valid_MAD, ite = [], [], [], [], []
        index_tra_samp = np.arange(num_train)
        index_val_samp = np.arange(num_validation)
        for ith_iteration in range(1, iteration + 1):
            input, target = self._load_batch_data(index_sample = index_tra_samp, data_tyep = 'training_set')
            input, target = input.to(device), target.to(device)

            for t in range(repeat_n):
                out = self._net(input)

                loss = self.loss_function(out, target,using_loss_fft = using_loss_fft)
                optimizer.zero_grad()  #
                loss.backward()  #
                optimizer.step()  #
            new_lr = self.LrDecay(iteration=iteration, ith_iteration=ith_iteration, lr=learning_rate)
            optimizer.param_groups[0]['lr'] = new_lr
            pbar.update(1)
            if ith_iteration % ns == 0:
                input_valid, target_valid = self._load_batch_data(index_sample = index_val_samp, data_tyep='validation_set')
                input_valid = input_valid.to(device)
                self._net.eval()
                output_valid = self._net(input_valid)
                output_valid = output_valid.data.cpu()
                val_loss = self.loss_function(output_valid, target_valid,using_loss_fft = using_loss_fft)
                ite.append(ith_iteration)
                train_loss.append(loss.item())
                valid_loss.append(val_loss.item())
                tra_MAD = self.MAD(out.data.cpu().numpy(), target.cpu().numpy())
                val_MAD = self.MAD(output_valid.numpy(), target_valid.numpy())
                train_MAD.append(tra_MAD)
                valid_MAD.append(val_MAD)
                self._net.train()
                print(
                    "Iteration: [%d/%d] training loss: %.6f validation_loss: %.6f mean MAD in train set: %.6f mean MAD in valid set: %.6f " %
                    (ith_iteration + 1, iteration, loss.item(), val_loss.item(), tra_MAD, val_MAD))

                torch.save(self._net, self.records_dir+'net_%s.pkl' % (ith_iteration))

                tra_log = {'iter': ite, 'train_loss': train_loss, 'valid_loss': valid_loss, 'train_MAD': train_MAD,
                          'valid_MAD': valid_MAD}
                utils.save_pkl(tra_log, self.records_dir + 'train_log_%s' % (ith_iteration))

        torch.save(self._net, self.records_dir + 'net.pkl')
        utils.save_pkl(tra_log, self.records_dir + 'train_log')


    def _predictor(self, input_test, target_test, net = None):
        net.eval()
        out = net(input_test)
        out = out.cpu()
        out = torch.squeeze(out, 0)
        out = torch.squeeze(out, 0).detach().numpy()
        target_test = torch.squeeze(target_test, 0)
        target_test = torch.squeeze(target_test, 0)
        target_test = target_test.detach().numpy()
        return out, target_test




    def plot_train_records(self, train_log = None, train_log_dir = None):
        if train_log is None:
            if train_log_dir is None:
                train_log_dir = self.records_dir + 'train_log'
            train_log = utils.load_pkl(train_log_dir)
        self.records_dir = getattr(self, 'records_{}map_dir'.format(self.component))
        iter, train_loss, valid_loss, train_MAD, valid_MAD  = train_log['iter'], train_log['train_loss'], train_log['valid_loss'], train_log['train_MAD'], train_log['valid_MAD']
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        ax1 = axs[0]
        ax1.scatter(iter, train_loss, c='r', s=3, lw=2)
        ax1.plot(iter, train_loss, c='r', lw=2, label='training loss')
        ax1.scatter(iter, valid_loss, c='c', s=3, lw=2)
        ax1.plot(iter, valid_loss, c='c', lw=2, label='validation loss')
        ax1.legend(loc='best', fontsize=16)
        ax1.set_xlabel('Iteration', fontsize=18)
        ax1.set_ylabel('Loss Function', fontsize=18)
        ax1.tick_params(labelsize=16)
        ax1.spines['bottom'].set_linewidth(1.5)
        ax1.spines['left'].set_linewidth(1.5)
        ax1.spines['top'].set_linewidth(1.5)
        ax1.spines['right'].set_linewidth(1.5)
        ax1.tick_params(which='major', length=8, direction='in', width=1.5)
        ax1.tick_params(which='minor', length=4, direction='in', width=1.5)
        ax1.tick_params(which='both', width=1.5, right='on')
        ax1.tick_params(which='both', width=1.5, top='on')

        ax2 = axs[1]
        ax2.scatter(iter, train_MAD, c='r', s=3, lw=2)
        ax2.plot(iter, train_MAD, c='r', lw=2, label='training MAD', )
        ax2.scatter(iter, valid_MAD, c='k', s=3, lw=2)
        ax2.plot(iter, valid_MAD, c='k', lw=2, label='validation MAD')
        ax2.legend(loc='best', fontsize=16)
        ax2.set_xlabel('Iteration', fontsize=18)
        ax2.set_ylabel('Mean Absolute Deviation', fontsize=18)
        ax2.tick_params(labelsize=16)
        ax2.spines['bottom'].set_linewidth(1.5)
        ax2.spines['left'].set_linewidth(1.5)
        ax2.spines['top'].set_linewidth(1.5)
        ax2.spines['right'].set_linewidth(1.5)
        # ax1.axis['bottom', 'left'].label.set_fontsize(16)
        # ax1.axis['bottom', 'left'].major_ticklabels.set_fontsize(16)
        ax2.tick_params(which='major', length=8, direction='in', width=1.5)
        ax2.tick_params(which='minor', length=4, direction='in', width=1.5)
        ax2.tick_params(which='both', width=1.5, right='on')
        ax2.tick_params(which='both', width=1.5, top='on')

        plt.savefig(self.records_dir+'plot_acc.png')
        # plt.show()


class Result_analysis(Foreground_subtraction_model):
    def __init__(self, data_dir='DATA/',  full_sky_map = False, map_block = 'block_0', padding = True, using_ilc_cmbmap=False,
                 is_half_split_map=True, result_dir = 'DATA_results/',freqs = np.array([]),  output_freq = 220, nside=512):
        if not data_dir is None:
            self._save_dir = data_dir
            self._cread_file_name
        self.result_dir = result_dir
        self.full_sky_map = full_sky_map
        self.is_half_split_map = is_half_split_map
        self.map_block  = map_block
        self.padding = padding
        self.freqs = freqs
        self.output_freq = output_freq
        self.using_ilc_cmbmap = using_ilc_cmbmap
        self.nside = nside
        self._creat_file_n


    def predictor(self, net_dir, num_testset=300, comp='Q'):

        net_dir = net_dir + 'net.pkl'
        net = torch.load(net_dir)
        if self.is_half_split_map:
            pbar = tqdm(total=num_testset, miniters=1)
            pbar.set_description('Predicting the CMB {} map'.format(comp))
            for n  in range(num_testset):

                input_test_1, target_test_1 = self._load_batch_data(index_sample=[n], data_tyep='testing_set', batch_size = 1, half_sp = 1, comp =comp)
                input_test_2, target_test_2 = self._load_batch_data(index_sample=[n], data_tyep='testing_set',
                                                                     batch_size=1, half_sp=2, comp = comp)
                if n==0:
                    out_1, target_1 = self._predictor(input_test_1, target_test_1, net)
                    out_2, target_2 = self._predictor(input_test_2, target_test_2, net)
                    out_1, target_1, out_2, target_2 = out_1[None,:], target_1[None,:], out_2[None,:], target_2[None,:]
                else:
                    out_1i, target_1i = self._predictor(input_test_1, target_test_1, net)
                    out_2i, target_2i = self._predictor(input_test_2, target_test_2, net)
                    out_1 = np.r_[out_1, out_1i[None,:]]
                    out_2 = np.r_[out_2, out_2i[None,:]]
                    target_1 = np.r_[target_1, target_1i[None,:]]
                    target_2 = np.r_[target_2, target_2i[None,:]]
                pbar.update(1)
            return out_1, out_2, target_1, target_2

        else:
            for n in range(num_testset):
                pbar = tqdm(total=num_testset, miniters=1)
                pbar.set_description('Predicting the CMB {} map'.format(comp))
                input_test, target_test = self._load_batch_data(index_sample=[n], data_tyep='testing_set',
                                                                 batch_size=1,  comp=comp)
                if n == 0:
                    out, target = self._predictor(input_test, target_test, net)
                    out, target = out[None,:], target[None,:]
                else:
                    out_i, target_i = self._predictor(input_test, target_test, net)
                    out = np.r_[out, out_i[None,:]]
                    target = np.r_[target, target_i[None,:]]
                pbar.update(1)
            return out, target

    def prediction(self, num_testset=300, comp = 'TQU', is_get_sphere_map = True, is_return = False):

        for str in comp:
            net_dir = getattr(self, 'records_{}map_dir'.format(str))
            output_dir = getattr(self, 'output_{}map_dir'.format(str))
            self._creat_file(output_dir)
            if self.is_half_split_map:
                out_1, out_2, target_1, target_2 = self.predictor(net_dir = net_dir, num_testset = num_testset, comp = str)
                np.save(output_dir + 'predicted_CMB_' + str + '_map_half_1.npy', (out_1).astype(np.float32))
                np.save(output_dir + 'predicted_CMB_' + str + '_map_half_2.npy', (out_2).astype(np.float32))
                np.save(output_dir + 'target_CMB_' + str + '_map_half_1.npy', (target_1).astype(np.float32))
                np.save(output_dir + 'target_CMB_' + str + '_map_half_2.npy', (target_2).astype(np.float32))
                if is_get_sphere_map:
                    out_s1 = self.sphere_from_flat_map(flat_map=out_1, is_fullsky = self.full_sky_map, is_padding = self.padding, block = self.map_block)
                    out_s2 = self.sphere_from_flat_map(flat_map=out_2, is_fullsky=self.full_sky_map,
                                                      is_padding=self.padding, block=self.map_block)
                    target_s1 = self.sphere_from_flat_map(flat_map=target_1, is_fullsky=self.full_sky_map,
                                                      is_padding=self.padding, block=self.map_block)
                    target_s2 = self.sphere_from_flat_map(flat_map=target_2, is_fullsky=self.full_sky_map,
                                                      is_padding=self.padding, block=self.map_block)
                    np.save(output_dir + 'predicted_sphere_CMB_' + str + '_map_half_1.npy', (out_s1).astype(np.float32))
                    np.save(output_dir + 'predicted_sphere_CMB_' + str + '_map_half_2.npy', (out_s2).astype(np.float32))
                    np.save(output_dir + 'target_sphere_CMB_' + str + '_map_half_1.npy', (target_s1).astype(np.float32))
                    np.save(output_dir + 'target_sphere_CMB_' + str + '_map_half_2.npy', (target_s2).astype(np.float32))
                if is_return:
                    if is_get_sphere_map:
                        return out_1, out_2, target_1, target_2, out_s1, out_s2, target_s1, target_s2
                    else:
                        return out_1, out_2, target_1, target_2
            else:
                out, target = self.predictor(net_dir=net_dir, num_testset=num_testset, comp=str)
                np.save(output_dir + 'predicted_CMB_' + str + '_map.npy', (out).astype(np.float32))
                np.save(output_dir + 'target_CMB_' + str + '_map.npy', (target).astype(np.float32))
                if is_get_sphere_map:
                    out_s = self.sphere_from_flat_map(flat_map=out, is_fullsky=self.full_sky_map,
                                                       is_padding=self.padding, block=self.map_block)
                    target_s = self.sphere_from_flat_map(flat_map=target, is_fullsky=self.full_sky_map,
                                                          is_padding=self.padding, block=self.map_block)
                    np.save(output_dir + 'predicted_sphere_CMB_' + str + '_map.npy', (out_s).astype(np.float32))
                    np.save(output_dir + 'target_sphere_CMB_' + str + '_map.npy', (target_s).astype(np.float32))
                if is_return:
                    if is_get_sphere_map:
                        return out, target, out_s, target_s
                    else:
                        return out, target

    def get_sphere_predicted_map(self, comp = 'TQU',num_testset=300):
        for str in comp:
            output_dir = getattr(self, 'output_{}map_dir'.format(str))
            self._creat_file(output_dir)
            try:
                out_1 = np.load(output_dir + 'predicted_CMB' + str + '_map_half_1.npy')
            except:
               self.prediction(self, num_testset=num_testset, comp = str, is_get_sphere_map = True, is_return = False)

    def get_true_CMB_map(self, index_sky_map):
        pbar = tqdm(total=len(index_sky_map), miniters=1)
        pbar.set_description('Geting the True CMB map')

        for n, i in enumerate(index_sky_map):
            if n==0:
                cmb = np.load(self.dir_cmb + 'cmb' + str(i) + '.npy')[self.output_freq_index,:]
                cmb = cmb[None,:]
            else:
                cmb_i = np.load(self.dir_cmb + 'cmb' + str(i) + '.npy')[self.output_freq_index,:]
                cmb = np.r_[cmb, cmb_i[None,:]]
            pbar.update(1)
        if cmb.shape[1] ==2:
            output_Qdir = getattr(self, 'output_Qmap_dir')
            output_Udir = getattr(self, 'output_Umap_dir')
            self._creat_file(output_Qdir)
            self._creat_file(output_Udir)
            np.save(output_Qdir + 'true_cmb_Q_map' + '.npy', (cmb[:,0,:]).astype(np.float32))
            np.save(output_Udir + 'true_cmb_U_map' + '.npy', (cmb[:,1, :]).astype(np.float32))
        else:
            output_Tdir = getattr(self, 'output_Tmap_dir')
            output_Qdir = getattr(self, 'output_Qmap_dir')
            output_Udir = getattr(self, 'output_Umap_dir')
            self._creat_file(output_Tdir)
            self._creat_file(output_Qdir)
            self._creat_file(output_Udir)
            np.save(output_Tdir + 'true_cmb_T_map' + '.npy', (cmb[:, 0, :]).astype(np.float32))
            np.save(output_Qdir + 'true_cmb_Q_map' + '.npy', (cmb[:, 1, :]).astype(np.float32))
            np.save(output_Udir + 'true_cmb_U_map' + '.npy', (cmb[:, 2, :]).astype(np.float32))




class Calculate_power_spectra(Result_analysis):
    def __init__(self,  result_dir = 'DATA_results/',is_half_split_map=True, component = 'Q',
                 beams_arcmin = np.array([]), output_beam_arcmin = 220, N_sample=10, nside = 512):
        # super(Calculate_power_spectra, self).__init__()
        self.is_half_split_map= is_half_split_map
        self.result_dir = result_dir
        self._creat_file_n
        self.beams = beams_arcmin
        self.output_beam = output_beam_arcmin
        self.N_sample = N_sample
        self.nside = nside
        self.component = component
        self._creat_file(self.save_PS_dir)
        self._creat_ps_file

    @property
    def save_PS_dir(self):
        return self.result_dir + 'output_PS/'

    @property
    def _creat_ps_file(self):
        dir_key = {}
        Tfile_dir = getattr(self, 'output_Tmap_dir')
        Qfile_dir = getattr(self, 'output_Qmap_dir')
        Ufile_dir = getattr(self, 'output_Umap_dir')
        dir_key['true_T_dir'] = Tfile_dir + 'true_cmb_T_map' + '.npy'
        dir_key['true_Q_dir'] = Qfile_dir + 'true_cmb_Q_map' + '.npy'
        dir_key['true_U_dir'] = Ufile_dir + 'true_cmb_U_map' + '.npy'
        dir_key['output_T_dir'] = Tfile_dir + 'predicted_sphere_CMB_T' + '_map.npy'
        dir_key['output_Q_dir'] = Qfile_dir + 'predicted_sphere_CMB_Q' + '_map.npy'
        dir_key['output_U_dir'] = Ufile_dir + 'predicted_sphere_CMB_U' + '_map.npy'
        dir_key['output_T_dir_1'] = Tfile_dir + 'predicted_sphere_CMB_T' + '_map_half_{}.npy'.format(1)
        dir_key['output_T_dir_2'] = Tfile_dir + 'predicted_sphere_CMB_T' + '_map_half_{}.npy'.format(2)
        dir_key['output_Q_dir_1'] = Qfile_dir + 'predicted_sphere_CMB_Q' + '_map_half_{}.npy'.format(1)
        dir_key['output_Q_dir_2'] = Qfile_dir + 'predicted_sphere_CMB_Q' + '_map_half_{}.npy'.format(2)
        dir_key['output_U_dir_1'] = Ufile_dir + 'predicted_sphere_CMB_U' + '_map_half_{}.npy'.format(1)
        dir_key['output_U_dir_2'] = Ufile_dir + 'predicted_sphere_CMB_U' + '_map_half_{}.npy'.format(2)
        dir_key['target_Q_dir'] = Qfile_dir + 'target_sphere_CMB_Q' + '_map.npy'
        dir_key['target_T_dir'] = Tfile_dir + 'target_sphere_CMB_T' + '_map.npy'
        dir_key['target_U_dir'] = Ufile_dir + 'target_sphere_CMB_U' + '_map.npy'
        dir_key['target_T_dir_1'] = Tfile_dir + 'target_sphere_CMB_T' + '_map_half_{}.npy'.format(1)
        dir_key['target_T_dir_2'] = Tfile_dir + 'target_sphere_CMB_T' + '_map_half_{}.npy'.format(2)
        dir_key['target_Q_dir_1'] = Qfile_dir + 'target_sphere_CMB_Q' + '_map_half_{}.npy'.format(1)
        dir_key['target_Q_dir_2'] = Qfile_dir + 'target_sphere_CMB_Q' + '_map_half_{}.npy'.format(2)
        dir_key['target_U_dir_1'] = Ufile_dir + 'target_sphere_CMB_U' + '_map_half_{}.npy'.format(1)
        dir_key['target_U_dir_2'] = Ufile_dir + 'target_sphere_CMB_U' + '_map_half_{}.npy'.format(2)

        dir_key['true_output_T_dir'] = self.save_PS_dir + 'true_cmb_T_PS_nlb_{}.npy'
        dir_key['true_output_Q_dir'] = self.save_PS_dir+'true_cmb_Q_PS_nlb_{}.npy'
        dir_key['true_output_U_dir'] = self.save_PS_dir + 'true_cmb_U_PS_nlb_{}.npy'

        dir_key['save_output_Q_dir_1'] = self.save_PS_dir + 'output_cmb_Q_PS_nlb_{}_half_1.npy'
        dir_key['save_output_Q_dir_2'] = self.save_PS_dir + 'output_cmb_Q_PS_nlb_{}_half_2.npy'
        dir_key['save_output_T_dir_1'] = self.save_PS_dir + 'output_cmb_T_PS_nlb_{}_half_1.npy'
        dir_key['save_output_T_dir_2'] = self.save_PS_dir + 'output_cmb_T_PS_nlb_{}_half_2.npy'
        dir_key['save_output_U_dir_1'] = self.save_PS_dir + 'output_cmb_U_PS_nlb_{}_half_1.npy'
        dir_key['save_output_U_dir_2'] = self.save_PS_dir + 'output_cmb_U_PS_nlb_{}_half_2.npy'

        dir_key['save_target_T_dir_1'] = self.save_PS_dir + 'target_cmb_T_PS_nlb_{}_half_1.npy'
        dir_key['save_target_T_dir_2'] = self.save_PS_dir + 'target_cmb_T_PS_nlb_{}_half_2.npy'
        dir_key['save_target_Q_dir_1'] = self.save_PS_dir + 'target_cmb_Q_PS_nlb_{}_half_1.npy'
        dir_key['save_target_Q_dir_2'] = self.save_PS_dir + 'target_cmb_Q_PS_nlb_{}_half_2.npy'
        dir_key['save_target_U_dir_1'] = self.save_PS_dir + 'target_cmb_U_PS_nlb_{}_half_1.npy'
        dir_key['save_target_U_dir_2'] = self.save_PS_dir + 'target_cmb_U_PS_nlb_{}_half_2.npy'

        dir_key['save_output_cros_T_dir'] = self.save_PS_dir + 'output_cmb_T_cross_PS_nlb_{}.npy'
        dir_key['save_output_cros_Q_dir'] = self.save_PS_dir + 'output_cmb_Q_cross_PS_nlb_{}.npy'
        dir_key['save_output_cros_U_dir'] = self.save_PS_dir + 'output_cmb_U_cross_PS_nlb_{}.npy'

        dir_key['save_output_EB_dir'] = self.save_PS_dir + 'output_cmb_EB_PS_nlb_{}.npy'
        dir_key['save_target_EB_dir'] = self.save_PS_dir + 'target_cmb_EB_PS_nlb_{}.npy'

        dir_key['save_true_EB_dir'] = self.save_PS_dir + 'true_cmb_EB_PS_nlb_{}.npy'

        dir_key['save_output_cros_EB_dir'] = self.save_PS_dir + 'output_cmb_EB_cross_PS_nlb_{}.npy'


        for key, value in dir_key.items():
            setattr(self, key, value)

    def __power_for_Nsample_from_spheremap(self,data_Q1_dir=None, data_Q2_dir = None,  data_U1_dir=None, data_U2_dir = None,
                                            nlb=5, Dl=True, aposize=None, EB_power=False):
        beam = utils.get_Bl(self.output_beam, nside=self.nside)

        Dl_gene = utils.Get_power(data_Q1_dir=data_Q1_dir, data_Q2_dir=data_Q2_dir, data_U1_dir=data_U1_dir, data_U2_dir=data_U2_dir, block_n=None, nside=self.nside,
                            beam_file=beam, nlb=nlb, Dl=Dl, aposize=aposize)
        Dl = Dl_gene.get_N_power_from_spheremap(N_sample=np.arange(self.N_sample), EB_power=EB_power)
        return Dl

    def get_true_CMB_PS(self, nlb=5, Dl=True, aposize=None, EB_power=False, half=1):
        if 'T' in self.component:
            Dl_true_T = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'true_T_dir'), nlb=nlb, Dl=Dl,
                                                                aposize=aposize, EB_power=False)

            np.save(getattr(self, 'true_output_T_dir').format(nlb), Dl_true_T)

        else:
            if EB_power:
                Dl_true_EB = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'true_Q_dir'), data_U1_dir=getattr(self, 'true_U_dir'),
                                                                     nlb=nlb, Dl=Dl,
                                                                    aposize=aposize, EB_power=EB_power)
                np.save(getattr(self, 'save_true_EB_dir').format(nlb), Dl_true_EB)
            else:
                Dl_true_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'true_Q_dir'), nlb=nlb, Dl=Dl,
                                                                    aposize=aposize, EB_power=EB_power)
                Dl_true_U = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'true_U_dir'), nlb=nlb, Dl=Dl,
                                                                    aposize=aposize,
                                                                    EB_power=EB_power)
                np.save(getattr(self, 'true_output_Q_dir').format(nlb), Dl_true_Q)
                np.save(getattr(self, 'true_output_U_dir').format(nlb), Dl_true_U)


    def cal_cmb_Q_or_U_PS(self, nlb=5, Dl=True, aposize=None, EB_power=False, half=1):
        Qfile_dir = getattr(self, 'output_Qmap_dir')
        Ufile_dir = getattr(self, 'output_Umap_dir')


        if self.is_half_split_map:
            output_Q_dir = Qfile_dir + 'predicted_sphere_CMB_Q' + '_map_half_{}.npy'.format(half)
            output_U_dir = Ufile_dir + 'predicted_sphere_CMB_U' + '_map_half_{}.npy'.format(half)
            target_Q_dir = Qfile_dir + 'target_sphere_CMB_Q' + '_map_half_{}.npy'.format(half)
            target_U_dir = Ufile_dir + 'target_sphere_CMB_U' + '_map_half_{}.npy'.format(half)

            save_output_Q_dir = self.save_PS_dir + 'output_cmb_Q_PS_nlb_{}_half_{}.npy'.format(nlb,half)
            save_output_U_dir = self.save_PS_dir + 'output_cmb_U_PS_nlb_{}_half_{}.npy'.format(nlb,half)
            save_target_Q_dir = self.save_PS_dir + 'target_cmb_Q_PS_nlb_{}_half_{}.npy'.format(nlb,half)
            save_target_U_dir = self.save_PS_dir + 'target_cmb_U_PS_nlb_{}_half_{}.npy'.format(nlb,half)

        else:
            output_Q_dir = Qfile_dir + 'predicted_sphere_CMB_Q' + '_map.npy'
            output_U_dir = Ufile_dir + 'predicted_sphere_CMB_U' + '_map.npy'
            target_Q_dir = Qfile_dir + 'target_sphere_CMB_Q' + '_map.npy'
            target_U_dir = Ufile_dir + 'target_sphere_CMB_U' + '_map.npy'

            save_output_Q_dir = self.save_PS_dir + 'output_cmb_Q_PS_nlb_{}.npy'.format(nlb)
            save_output_U_dir = self.save_PS_dir + 'output_cmb_U_PS_nlb_{}.npy'.format(nlb)
            save_target_Q_dir = self.save_PS_dir + 'target_cmb_Q_PS_nlb_{}.npy'.format(nlb)
            save_target_U_dir = self.save_PS_dir + 'target_cmb_U_PS_nlb_{}.npy'.format(nlb)

        Dl_output_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=output_Q_dir, nlb=nlb, Dl=Dl, aposize=aposize,
                                                            EB_power=EB_power)
        Dl_output_U = self.__power_for_Nsample_from_spheremap(data_Q1_dir=output_U_dir, nlb=nlb, Dl=Dl,
                                                              aposize=aposize,
                                                              EB_power=EB_power)
        Dl_target_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=target_Q_dir, nlb=nlb, Dl=Dl,
                                                              aposize=aposize,
                                                              EB_power=EB_power)
        Dl_target_U = self.__power_for_Nsample_from_spheremap(data_Q1_dir=target_U_dir, nlb=nlb, Dl=Dl,
                                                              aposize=aposize,
                                                              EB_power=EB_power)
        np.save(save_output_Q_dir, Dl_output_Q)
        np.save(save_output_U_dir, Dl_output_U)
        np.save(save_target_Q_dir, Dl_target_Q)
        np.save(save_target_U_dir, Dl_target_U)

    def cal_cmb_T_PS(self, nlb=5, Dl=True, aposize=None, EB_power=False, half=1):
        Qfile_dir = getattr(self, 'output_Tmap_dir')

        if self.is_half_split_map:
            output_T_dir = Qfile_dir + 'predicted_sphere_CMB_T' + '_map_half_{}.npy'.format(half)
            target_T_dir = Qfile_dir + 'target_sphere_CMB_T' + '_map_half_{}.npy'.format(half)

            save_output_T_dir = self.save_PS_dir + 'output_cmb_T_PS_nlb_{}_half_{}.npy'.format(nlb,half)
            save_target_T_dir = self.save_PS_dir + 'target_cmb_T_PS_nlb_{}_half_{}.npy'.format(nlb,half)

        else:
            output_T_dir = Qfile_dir + 'predicted_sphere_CMB_T' + '_map.npy'
            target_T_dir = Qfile_dir + 'target_sphere_CMB_T' + '_map.npy'

            save_output_T_dir = self.save_PS_dir + 'output_cmb_T_PS_nlb_{}.npy'.format(nlb)
            save_target_T_dir = self.save_PS_dir + 'target_cmb_T_PS_nlb_{}.npy'.format(nlb)

        Dl_output_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=output_T_dir, nlb=nlb, Dl=Dl, aposize=aposize,
                                                            EB_power=EB_power)

        Dl_target_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=target_T_dir, nlb=nlb, Dl=Dl,
                                                              aposize=aposize,
                                                              EB_power=EB_power)
        np.save(save_output_T_dir, Dl_output_Q)
        np.save(save_target_T_dir, Dl_target_Q)

    def cal_cmb_T_cross_PS(self, nlb=5, Dl=True, aposize=None, EB_power=False):

        Dl_output_T = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self,'output_T_dir_1'), data_Q2_dir=getattr(self,'output_T_dir_2'), nlb=nlb, Dl=Dl, aposize=aposize,
                                                              EB_power=EB_power)

        np.save(getattr(self,'save_output_cros_T_dir').format(nlb), Dl_output_T)



    def cal_cmb_Q_or_U_cross_PS(self, nlb=5, Dl=True, aposize=None, EB_power=False):

        Dl_output_Q = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self,'output_Q_dir_1'), data_Q2_dir=getattr(self,'output_Q_dir_2'), nlb=nlb, Dl=Dl, aposize=aposize,
                                                              EB_power=EB_power)
        Dl_output_U = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self,'output_U_dir_1'), data_Q2_dir=getattr(self,'output_U_dir_2'), nlb=nlb, Dl=Dl,
                                                              aposize=aposize,
                                                              EB_power=EB_power)
        np.save(getattr(self,'save_output_cros_Q_dir').format(nlb), Dl_output_Q)
        np.save(getattr(self,'save_output_cros_U_dir').format(nlb), Dl_output_U)

    def cal_cmb_E_B_PS(self, nlb=5, Dl=True, aposize=None, EB_power=True):

        Dl_output_EB = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'output_Q_dir_1'),
                                                               data_U1_dir=getattr(self, 'output_U_dir_1'),nlb=nlb,
                                                              Dl=Dl, aposize=aposize,
                                                              EB_power=EB_power)
        Dl_tar_EB = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'target_Q_dir_1'),
                                                               data_U1_dir=getattr(self, 'target_U_dir_1'), nlb=nlb,
                                                               Dl=Dl, aposize=aposize,
                                                               EB_power=EB_power)

        np.save(getattr(self,'save_output_EB_dir').format(nlb), Dl_output_EB)
        np.save(getattr(self,'save_target_EB_dir').format(nlb), Dl_tar_EB)

    def cal_cmb_E_B_cross_PS(self, nlb=5, Dl=True, aposize=None, EB_power=True):

        Dl_output_EB = self.__power_for_Nsample_from_spheremap(data_Q1_dir=getattr(self, 'output_Q_dir_1'),
                                                              data_Q2_dir=getattr(self, 'output_Q_dir_2'), data_U1_dir=getattr(self, 'output_U_dir_1'),
                                                              data_U2_dir=getattr(self, 'output_U_dir_2'),nlb=nlb,
                                                              Dl=Dl, aposize=aposize,
                                                              EB_power=EB_power)

        np.save(getattr(self,'save_output_cros_EB_dir').format(nlb), Dl_output_EB)





































