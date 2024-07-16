
# -*- coding: utf-8 -*-

from . import simulate_sky_map as skm
import numpy as np
import os,copy
from tqdm import tqdm
import multiprocessing
from . import spherical as sp
import torch
import healpy as hp
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

train_config_random_one_05 = {'syn_spectralindex_random':(0.05, 'one'), 'syn_amplitude_random':(0, 'one'),
                           'dust_spectralindex_random':(0.05, 'one'), 'dust_amplitude_random':(0., 'one'),
                           'dust_temp_random':(0.05, 'multi'),
                               'ame_amplitude_random':(0, 'one'), 'Random_types_of_cosmological_parameters':'Uniform'}


class Simulator_data(object):
    def __init__(self, freqs=None, beams=None, noise_level = None, Nside=512, is_half_split_map=True,
                 config_random=train_config_random_one_05, out_unit = 'uK_CMB', save_dir='DATA/',
                 using_beam = True, is_polarization_data=True, using_ilc_cmbmap=False):
        '''
        Simulate multi band sky signals and noise
        :param freqs: The frequencies of the instrument
        :param beams: The beam of the instrument. FWHM (arcmin)
        :param noise_level: White noise level (uK-arcmin)
        :param Nside: Nside for healpix map
        :param config_random: A configuration dictionary with random parameters. These parameters include foreground model parameters and cosmological parameters.
        :param out_unit: Output units (uK_CMB)
        :param save_dir: Directory where data is saved
        :param using_beam: Add beam to sky maps
        '''
        self.freqs = freqs
        self.beams = beams
        self.noise_level = noise_level
        self.nside = Nside
        self.config_random = config_random
        self.out_unit = out_unit
        self._save_dir = save_dir
        self.using_beam = using_beam
        self.is_half_split_map = is_half_split_map
        self.is_polarization_data = is_polarization_data
        self.using_ilc_cmbmap = using_ilc_cmbmap
        self._cread_file_name

    @property
    def save_dir(self):
        return self._save_dir

    def __get_map(self):
        data = self.__get_Data()
        cmb, total = data.data()  # total represents CMB + foregrounds
        if self.is_polarization_data:
            cmb = cmb[:,1:,:]
            total = total[:,1:,:]
        return cmb, total

    @property
    def Config_random(self):
        return self.config_random

    def __get_noise(self):
        data = self.__get_Data()
        if self.noise_level is None:
            print('An error occurred: Please provide the level of white noise')
        noise = data.noiser(Sens = self.noise_level, is_half_split_map = self.is_half_split_map)
        if self.is_polarization_data:
            noise = noise[:,1:,:]
        return noise

    @property
    def _cread_file_name(self):
        self.dir_cmb = self.save_dir + 'noiseless/cmb/'
        self.dir_total = self.save_dir + 'noiseless/total/'
        dir_key = {}
        for ty in ['training_set', 'validation_set', 'testing_set']:

            dir_key['dir_noise_' + ty[0:3]] = self.save_dir + 'noise/{}/'.format(ty)
            dir_key['dir_noise_ILC_' + ty[0:3]] = self.save_dir + 'noise/{}_ILC/'.format(ty)
            dir_key['dir_noise_{}_reshape'.format(ty[0:3])] = self.save_dir + 'noise/{}_reshape/'.format(ty)
            dir_key['dir_cmb_obs_' + ty[0:3]] = self.save_dir + 'observed/{}/cmb/'.format(ty)
            dir_key['dir_cmb_ilc_' + ty[0:3]] = self.save_dir + 'observed/{}/cmb_ilcnoise/'.format(ty)
            dir_key['dir_total_obs_' + ty[0:3]] = self.save_dir + 'observed/{}/total/'.format(ty)
            dir_key['dir_cmb_obs_flat_' + ty[0:3]] = self.save_dir + 'observed_flat_map/{}/cmb/'.format(ty)
            dir_key['dir_cmb_obs_flat_block_' + ty[0:3]] = self.save_dir + 'observed_flat_block_map/{}/cmb/'.format(ty)
            dir_key['dir_total_obs_flat_' + ty[0:3]] = self.save_dir + 'observed_flat_map/{}/total/'.format(ty)
            dir_key['dir_total_obs_flat_block_' + ty[0:3]] = self.save_dir + 'observed_flat_block_map/{}/total/'.format(ty)
            dir_key['dir_cmb_ilc_flat_{}'.format(
                ty[0:3])] = self.save_dir + 'observed_flat_map/{}/cmb_ilcnoise/'.format(ty)
            dir_key['dir_cmb_ilc_flat_block_{}'.format(
                ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/cmb_ilcnoise/'.format(ty)
            if not ty == 'training_set':
                dir_key['dir_noise_ILC_{}_1'.format(ty[0:3])] = self.save_dir + 'noise/{}_ILC/half_1/'.format(ty)
                dir_key['dir_noise_ILC_{}_2'.format(ty[0:3])] = self.save_dir + 'noise/{}_ILC/half_2/'.format(ty)
                dir_key['dir_noise_{}_1'.format(ty[0:3])] = self.save_dir + 'noise/{}/half_1/'.format(ty)
                dir_key['dir_noise_{}_2'.format(ty[0:3])] = self.save_dir + 'noise/{}/half_2/'.format(ty)
                dir_key['dir_cmb_obs_{}_1'.format(ty[0:3])] = self.save_dir + 'observed/{}/cmb/half_1/'.format(ty)
                dir_key['dir_cmb_ilc_{}_1'.format(ty[0:3])] = self.save_dir + 'observed/{}/cmb_ilcnoise/half_1/'.format(ty)
                dir_key['dir_total_obs_{}_1'.format(ty[0:3])] = self.save_dir + 'observed/{}/total/half_1/'.format(ty)
                dir_key['dir_cmb_obs_{}_2'.format(ty[0:3])] = self.save_dir + 'observed/{}/cmb/half_2/'.format(ty)
                dir_key['dir_cmb_ilc_{}_2'.format(ty[0:3])] = self.save_dir + 'observed/{}/cmb_ilcnoise/half_2/'.format(
                    ty)
                dir_key['dir_total_obs_{}_2'.format(ty[0:3])] = self.save_dir + 'observed/{}/total/half_2/'.format(ty)
                dir_key['dir_cmb_obs_flat_{}_1'.format(ty[0:3])] = self.save_dir + 'observed_flat_map/{}/cmb/half_1/'.format(ty)
                dir_key['dir_cmb_ilc_flat_{}_1'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_map/{}/cmb_ilcnoise/half_1/'.format(ty)
                dir_key['dir_total_obs_flat_{}_1'.format(ty[0:3])] = self.save_dir + 'observed_flat_map/{}/total/half_1/'.format(ty)
                dir_key['dir_cmb_obs_flat_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_map/{}/cmb/half_2/'.format(ty)
                dir_key['dir_cmb_ilc_flat_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_map/{}/cmb_ilcnoise/half_2/'.format(ty)
                dir_key['dir_total_obs_flat_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_map/{}/total/half_2/'.format(ty)
                dir_key['dir_cmb_obs_flat_block_{}_1'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/cmb/half_1/'.format(ty)
                dir_key['dir_cmb_ilc_flat_block_{}_1'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/cmb_ilcnoise/half_1/'.format(ty)
                dir_key['dir_total_obs_flat_block_{}_1'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/total/half_1/'.format(ty)
                dir_key['dir_cmb_obs_flat_block_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/cmb/half_2/'.format(ty)
                dir_key['dir_cmb_ilc_flat_block_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/cmb_ilcnoise/half_2/'.format(ty)
                dir_key['dir_total_obs_flat_block_{}_2'.format(
                    ty[0:3])] = self.save_dir + 'observed_flat_block_map/{}/total/half_2/'.format(ty)

        for key, value in dir_key.items():
            setattr(self, key, value)


    def __get_Data(self):
        return skm.Get_data(Nside=self.nside, freqs=self.freqs, out_unit=self.out_unit, config_random=self.Config_random,
                           using_beam=self.using_beam, beam=self.beams)
    def _creat_file(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def simulator_sky_map(self, N_sky_map):
        '''
        :param N_sky_map: The number of sky maps. An integer
        '''
        self._creat_file(self.dir_cmb)
        self._creat_file(self.dir_total)
        part_n = 11
        part_size = 110
        Dustseed = np.random.choice(N_sky_map*100000, N_sky_map*1000, replace=False)
        AMEseed = np.random.choice(N_sky_map*100000, N_sky_map*1000, replace=False)
        Syncseed = np.random.choice(N_sky_map*100000, N_sky_map*1000, replace=False)
        CMBseed = np.random.choice(N_sky_map*1000000, N_sky_map*1000, replace=False)
        pbar = tqdm(total = N_sky_map)
        pbar.set_description('Simulating sky signals')
        for n in range(N_sky_map):
            self.Config_random['cmb_seed'] = CMBseed[n+ part_n * part_size]
            self.Config_random['dust_seed'] = Dustseed[n + part_n * part_size]
            self.Config_random['syn_seed'] = Syncseed[n + part_n * part_size]
            self.Config_random['ame_seed'] = AMEseed[n + part_n * part_size]
            cmb, total = self.__get_map()
            np.save(self.dir_cmb+ 'cmb' + str(n) + '.npy', cmb.astype(np.float32))
            np.save(self.dir_total + 'total' + str(n) + '.npy', total.astype(np.float32))

            pbar.update(1)

    def simulator_noise_map(self, N_noise_map,  dataset_type = 'traing_set'):
        '''
        :param N_noise_map: The number of noise maps. An integer
        :param is_half_split_map: If it is an half-split map, the noise level will increase by sqrt(2) times
        :param dataset_type: Noise in training, validation, and testing sets. dataset_type = 'traing_set' or 'validation_set' or 'testing_set'
        '''
        if dataset_type == 'traing_set':
            self._creat_file(self.dir_noise_tra)

        elif dataset_type == 'validation_set':
            if self.is_half_split_map:
                self._creat_file(self.dir_noise_val_1)
                self._creat_file(self.dir_noise_val_2)
            else:
                self._creat_file(self.dir_noise_val)

        elif dataset_type == 'testing_set':
            if self.is_half_split_map:
                self._creat_file(self.dir_noise_tes_1)
                self._creat_file(self.dir_noise_tes_2)
            else:
                self._creat_file(self.dir_noise_tes)

        else:
            print('An error occurred: Please set the dataset_type correctly. dataset_type = "traing_set" or "validation_set" or "testing_set"')

        pbar = tqdm(total = N_noise_map, miniters=2)
        pbar.set_description('Simulating white noise')
        for nn in range(N_noise_map):
            if dataset_type == 'traing_set':
                noise = self.__get_noise()
                np.save(self.dir_noise_tra + 'noise' + str(nn) + '.npy', noise.astype(np.float32))

            elif dataset_type == 'validation_set':
                if self.is_half_split_map:
                    noise_1 = self.__get_noise()
                    np.save(self.dir_noise_val_1 + 'noise' + str(nn) + '.npy', noise_1.astype(np.float32))
                    noise_2 = self.__get_noise()
                    np.save(self.dir_noise_val_2 + 'noise'  + str(nn) + '.npy', noise_2.astype(np.float32))
                else:
                    noise_1 = self.__get_noise()
                    np.save(self.dir_noise_val + 'noise' + str(nn) + '.npy', noise_1.astype(np.float32))
            elif dataset_type == 'testing_set':
                if self.is_half_split_map:
                    noise_1 = self.__get_noise()
                    np.save(self.dir_noise_tes_1 + 'noise' + str(nn) + '.npy', noise_1.astype(np.float32))
                    noise_2 = self.__get_noise()
                    np.save(self.dir_noise_tes_2 + 'noise' + str(nn) + '.npy', noise_2.astype(np.float32))
                else:
                    noise_1 = self.__get_noise()
                    np.save(self.dir_noise_tes + 'noise' + str(nn) + '.npy', noise_1.astype(np.float32))
            pbar.update(1)


    def __map_index(self, N_sky_map, N_noise):
        if len(N_sky_map)>len(N_noise):
            noise_idex = np.random.choice(N_noise, len(N_sky_map), replace=True)
        else:
            noise_idex = N_noise
        return N_sky_map, noise_idex

    def get_observed_map(self, index_sky_map, index_noise_map, dataset_type = 'traing_set'):
        '''
        Add instrument noise to the sky maps
        :param index_sky_map: Index of sky map
        :param index_noise_map: Index of noise
        :param dataset_type: Noise in training, validation, and testing sets. dataset_type = 'traing_set' or 'validation_set' or 'testing_set'
        :return:
        '''
        index_sky_map, index_noise_map = self.__map_index(index_sky_map, index_noise_map)
        pbar = tqdm(total=len(index_sky_map), miniters=2)
        pbar.set_description('Obtaining observation sky map')
        for idx, _ in enumerate(zip(index_sky_map, index_noise_map)):
            k, n = _
            cmb = np.load(self.dir_cmb+ 'cmb' + str(k) + '.npy')
            total = np.load(self.dir_total + 'total' + str(k) + '.npy')
            if dataset_type == 'traing_set':
                self._creat_file(self.dir_cmb_obs_tra)
                self._creat_file(self.dir_total_obs_tra)
                self._creat_file(self.dir_noise_tra_reshape)
                noise = np.load(self.dir_noise_tra + 'noise' + str(n) + '.npy')
                np.save(self.dir_cmb_obs_tra + 'cmb' + str(idx) + '.npy', (cmb+noise).astype(np.float32))
                np.save(self.dir_total_obs_tra + 'total' + str(idx) + '.npy', (total+noise).astype(np.float32))
                np.save(self.dir_noise_tra_reshape+'noise'+ str(idx) + '.npy', (noise).astype(np.float32))

            elif dataset_type == 'validation_set':
                if self.is_half_split_map:
                    noise1 = np.load(self.dir_noise_val_1 + 'noise' + str(n) + '.npy')
                    noise2 = np.load(self.dir_noise_val_2 + 'noise' + str(n) + '.npy')
                    self._creat_file(self.dir_cmb_obs_val_1)
                    self._creat_file(self.dir_cmb_obs_val_2)
                    self._creat_file(self.dir_total_obs_val_1)
                    self._creat_file(self.dir_total_obs_val_2)
                    np.save(self.dir_cmb_obs_val_1 + 'cmb' + str(idx) + '.npy', (cmb + noise1).astype(np.float32))
                    np.save(self.dir_cmb_obs_val_2 + 'cmb' + str(idx) + '.npy', (cmb + noise2).astype(np.float32))
                    np.save(self.dir_total_obs_val_1 + 'total' + str(idx) + '.npy', (total + noise1).astype(np.float32))
                    np.save(self.dir_total_obs_val_2 + 'total' + str(idx) + '.npy', (total + noise2).astype(np.float32))
                else:
                    noise1 = np.load(self.dir_noise_val + 'noise' + str(n) + '.npy')
                    np.save(self.dir_cmb_obs_val + 'cmb' + str(idx) + '.npy', (cmb + noise1).astype(np.float32))
                    np.save(self.dir_total_obs_val + 'total' + str(idx) + '.npy', (total + noise1).astype(np.float32))

            elif dataset_type == 'testing_set':
                if self.is_half_split_map:
                    noise1 = np.load(self.dir_noise_tes_1 + 'noise' + str(n) + '.npy')
                    noise2 = np.load(self.dir_noise_tes_2 + 'noise' + str(n) + '.npy')
                    self._creat_file(self.dir_cmb_obs_tes_1)
                    self._creat_file(self.dir_cmb_obs_tes_2)
                    self._creat_file(self.dir_total_obs_tes_1)
                    self._creat_file(self.dir_total_obs_tes_2)
                    np.save(self.dir_cmb_obs_tes_1 + 'cmb' + str(idx) + '.npy', (cmb + noise1).astype(np.float32))
                    np.save(self.dir_cmb_obs_tes_2 + 'cmb' + str(idx) + '.npy', (cmb + noise2).astype(np.float32))
                    np.save(self.dir_total_obs_tes_1 + 'total' + str(idx) + '.npy', (total + noise1).astype(np.float32))
                    np.save(self.dir_total_obs_tes_2 + 'total' + str(idx) + '.npy', (total + noise2).astype(np.float32))
                else:
                    noise1 = np.load(self.dir_noise_tes + 'noise' + str(n) + '.npy')
                    np.save(self.dir_cmb_obs_tes + 'cmb' + str(idx) + '.npy', (cmb + noise1).astype(np.float32))
                    np.save(self.dir_total_obs_tes + 'total' + str(idx) + '.npy', (total + noise1).astype(np.float32))
            pbar.update(1)


    def run_ilc(self,N_samp, out_freq, mask=None, dataset_type = 'traing_set'):
        self.output_freq_index = np.where(self.freqs == out_freq)[0][0]
        import Utils_ILC as uilc
        beam = self.beams.tolist()
        bands = self.freqs.tolist()
        beam_profile = {}
        for k, v in zip(bands, beam):
            beam_profile['beam_' + str(k)] = hp.gauss_beam(v * np.pi / 10800., lmax=self.nside * 3, pol=False)
        shape = (len(bands), 3, hp.nside2npix(self.nside))
        maps = np.zeros(shape)
        if mask is None:
            mask = np.ones(12*self.nside**2)
        beams = np.array([beam_profile['beam_' + str(x)] for x in bands])
        outbeam = beam_profile['beam_' + str(out_freq)]
        def __ilc(data_qu, noise):
            if not self.is_polarization_data:
                data_qu = data_qu[:, 1:, :]
                noise_ = noise[:, 1:, :]
            else:
                noise_ = noise
            maps[:, 1:, :] = data_qu
            ilchandler = uilc.PILC(maps, beams, outbeam, mask=mask)
            resmap = ilchandler.do_ilc()
            weights = copy.deepcopy(ilchandler.weights)  # get weights
            del (ilchandler)

            maps[:, 1:, :] = noise_
            ilchandler = uilc.PILC(maps, beams, outbeam, mask=mask)
            ilchandler.set_weight(weights)  # set weights
            resmap = ilchandler.do_ilc()
            resmap *= mask
            del (ilchandler)
            return resmap

        pbar = tqdm(total=len(N_samp))
        pbar.set_description('Executing ILC')
        for idx in N_samp:
            if dataset_type == 'traing_set':
                data_qu = np.load(self.dir_total_obs_tra + 'total' + str(idx) + '.npy')
                noise = np.load(self.dir_noise_tra_reshape+'noise'+ str(idx) + '.npy')
                resmap = __ilc(data_qu, noise)
                if not self.is_polarization_data:
                    _maps = noise[self.output_freq_index,:].copy()
                    _maps[1:,:] = resmap
                else:
                    _maps = resmap

                self._creat_file(self.dir_noise_ILC_tra)
                np.save(self.dir_noise_ILC_tra+'ilcnoise{}.npy'.format(idx), _maps.astype(np.float32))
            elif dataset_type == 'validation_set':
                if self.is_half_split_map:
                    data_qu_1 = np.load(self.dir_total_obs_val_1 + 'total' + str(idx) + '.npy')
                    data_qu_2 = np.load(self.dir_total_obs_val_2 + 'total' + str(idx) + '.npy')
                    noise_1 = np.load(self.dir_noise_val_1 + 'noise' + str(idx) + '.npy')
                    noise_2 = np.load(self.dir_noise_val_2 + 'noise' + str(idx) + '.npy')
                    resmap_1 = __ilc(data_qu_1, noise_1)
                    resmap_2 = __ilc(data_qu_2, noise_2)
                    if not self.is_polarization_data:
                        _maps_1 = noise_1[self.output_freq_index, :].copy()
                        _maps_2 = noise_1[self.output_freq_index, :].copy()
                        _maps_1[1:, :] = resmap_1
                        _maps_1[1:, :] = resmap_2
                    else:
                        _maps_1 = resmap_1
                        _maps_2 = resmap_2
                    self._creat_file(self.dir_noise_ILC_val_1)
                    self._creat_file(self.dir_noise_ILC_val_2)
                    np.save(self.dir_noise_ILC_val_1 + 'ilcnoise{}.npy'.format(idx), _maps_1.astype(np.float32))
                    np.save(self.dir_noise_ILC_val_2 + 'ilcnoise{}.npy'.format(idx), _maps_2.astype(np.float32))
                else:
                    data_qu = np.load(self.dir_total_obs_val + 'total' + str(idx) + '.npy')
                    noise = np.load(self.dir_noise_val + 'noise' + str(idx) + '.npy')
                    resmap = __ilc(data_qu, noise)
                    if not self.is_polarization_data:
                        _maps = noise[self.output_freq_index, :].copy()
                        _maps[1:, :] = resmap
                    else:
                        _maps = resmap
                    self._creat_file(self.dir_noise_ILC_val)
                    np.save(self.dir_noise_ILC_val + 'ilcnoise{}.npy'.format(idx), _maps.astype(np.float32))

            elif dataset_type == 'testing_set':
                if self.is_half_split_map:
                    data_qu_1 = np.load(self.dir_total_obs_tes_1 + 'total' + str(idx) + '.npy')
                    data_qu_2 = np.load(self.dir_total_obs_tes_2 + 'total' + str(idx) + '.npy')
                    noise_1 = np.load(self.dir_noise_tes_1 + 'noise' + str(idx) + '.npy')
                    noise_2 = np.load(self.dir_noise_tes_2 + 'noise' + str(idx) + '.npy')
                    resmap_1 = __ilc(data_qu_1, noise_1)
                    resmap_2 = __ilc(data_qu_2, noise_2)
                    if not self.is_polarization_data:
                        _maps_1 = noise_1[self.output_freq_index, :].copy()
                        _maps_2 = noise_1[self.output_freq_index, :].copy()
                        _maps_1[1:, :] = resmap_1
                        _maps_1[1:, :] = resmap_2
                    else:
                        _maps_1 = resmap_1
                        _maps_2 = resmap_2
                    self._creat_file(self.dir_noise_ILC_tes_1)
                    self._creat_file(self.dir_noise_ILC_tes_2)
                    np.save(self.dir_noise_ILC_tes_1 + 'ilcnoise{}.npy'.format(idx), _maps_1.astype(np.float32))
                    np.save(self.dir_noise_ILC_tes_2 + 'ilcnoise{}.npy'.format(idx), _maps_2.astype(np.float32))
                else:
                    data_qu = np.load(self.dir_total_obs_tes + 'total' + str(idx) + '.npy')
                    noise = np.load(self.dir_noise_tes + 'noise' + str(idx) + '.npy')
                    resmap = __ilc(data_qu, noise)
                    if not self.is_polarization_data:
                        _maps = noise[self.output_freq_index, :].copy()
                        _maps[1:, :] = resmap
                    else:
                        _maps = resmap
                    self._creat_file(self.dir_noise_ILC_tes)
                    np.save(self.dir_noise_ILC_tes + 'ilcnoise{}.npy'.format(idx), _maps.astype(np.float32))


            pbar.update(1)

    def mult_process_get_ilc_noise(self, N_sample, N_mult=10, out_freq=220, mask=None, dataset_type = 'traing_set'):
        '''
        Multi threaded processing of ILC
        :param N_mult: Number of threads
        :param N_train_sample: the number of  map
        :return:
        '''

        N_per = N_sample // N_mult
        N_lop = N_sample // N_per
        N_rem = N_sample % N_per
        S_N = [np.arange(i * N_per, (i + 1) * N_per) for i in range(N_lop)]
        if not N_rem == 0:
            S_N_1 = [np.arange(N_lop * N_per, N_sample)]
            S_N.extend(S_N_1)


        processes = [multiprocessing.Process(target=self.run_ilc, kwargs={'N_samp': S_N[i], 'out_freq': out_freq, 'mask': mask, 'dataset_type': dataset_type}) for i in range(len(S_N))]
        for process in processes:
            process.start()
            process.join()

    def get_data_CMB_ilcnoise(self,index_sky_map, index_ilcnoise_map, dataset_type = 'traing_set', out_beam=166):
        self.output_freq_index = np.where(self.beams == out_beam)[0][0]
        pbar = tqdm(total=len(index_sky_map), miniters=2)
        pbar.set_description('Adding ILC noise')
        for idx, _ in enumerate(zip(index_sky_map, index_ilcnoise_map)):
            k, n = _
            cmb = np.load(self.dir_cmb+ 'cmb' + str(k) + '.npy')[self.output_freq_index, :]
            # if not self.is_polarization_data:
            #     cmb = cmb[1:, :]
            if dataset_type == 'traing_set':
                noise = np.load(self.dir_noise_ILC_tra+'ilcnoise{}.npy'.format(n))
                self._creat_file(self.dir_cmb_ilc_tra)
                np.save(self.dir_cmb_ilc_tra+ 'cmb' + str(idx) + '.npy', (cmb + noise).astype(np.float32))
            elif dataset_type == 'validation_set':
                if self.is_half_split_map:
                    noise_1 = np.load(self.dir_noise_ILC_val_1+'ilcnoise{}.npy'.format(n))
                    noise_2 = np.load(self.dir_noise_ILC_val_2 + 'ilcnoise{}.npy'.format(n))
                    self._creat_file(self.dir_cmb_ilc_val_1)
                    self._creat_file(self.dir_cmb_ilc_val_2)
                    np.save(self.dir_cmb_ilc_val_1 + 'cmb' + str(idx) + '.npy', (cmb + noise_1).astype(np.float32))
                    np.save(self.dir_cmb_ilc_val_2 + 'cmb' + str(idx) + '.npy', (cmb + noise_2).astype(np.float32))
                else:
                    noise = np.load(self.dir_noise_ILC_val + 'ilcnoise{}.npy'.format(n))
                    self._creat_file(self.dir_cmb_ilc_val)
                    np.save(self.dir_cmb_ilc_val + 'cmb' + str(idx) + '.npy', (cmb + noise).astype(np.float32))
            elif dataset_type == 'testing_set':
                if self.is_half_split_map:
                    noise_1 = np.load(self.dir_noise_ILC_tes_1+'ilcnoise{}.npy'.format(n))
                    noise_2 = np.load(self.dir_noise_ILC_tes_2 + 'ilcnoise{}.npy'.format(n))
                    self._creat_file(self.dir_cmb_ilc_tes_1)
                    self._creat_file(self.dir_cmb_ilc_tes_2)
                    np.save(self.dir_cmb_ilc_tes_1 + 'cmb' + str(idx) + '.npy', (cmb + noise_1).astype(np.float32))
                    np.save(self.dir_cmb_ilc_tes_2 + 'cmb' + str(idx) + '.npy', (cmb + noise_2).astype(np.float32))
                else:
                    noise = np.load(self.dir_noise_ILC_tes + 'ilcnoise{}.npy'.format(n))
                    self._creat_file(self.dir_cmb_ilc_tes)
                    np.save(self.dir_cmb_ilc_tes + 'cmb' + str(idx) + '.npy', (cmb + noise).astype(np.float32))
            pbar.update(1)



class Data_preprocessing(Simulator_data):
    '''
    Convert the spherical sky map to a planar sky map
    Convert a flat sky map to a spherical sky map
    '''
    def __init__(self, save_dir=None, padding=True, is_half_split_map = True, block_number='block_0',full_sky_map = True, nside=512, using_ilc_cmbmap=False):
        super(Data_preprocessing, self).__init__()

        self._save_dir = save_dir
        self.padding = padding
        self.block = block_number
        self.is_half_split_map = is_half_split_map
        self.full_sky_map =full_sky_map
        self.using_ilc_cmbmap = using_ilc_cmbmap
        self.nside = nside
        self._cread_file_name



    def flat_map_from_sphere_fullsky(self, sphere_map):

        if len(sphere_map.shape) == 3:
            freqs, comp_p, N_pix = sphere_map.shape
            map_new = np.ones((freqs, comp_p, 5 * self.nside, 5 * self.nside))
            for j in range(comp_p):
                map_new[:, j, :, :] = sp.sphere2piecePlane_squa_mult(sphere_map=sphere_map[:, j, :], nside=self.nside)
        else:
            map_new = sp.sphere2piecePlane_squa_mult(sphere_map=sphere_map, nside=self.nside)
        return map_new

    def flat_map_from_sphere_fullsky_padding(self, sphere_map):

        if len(sphere_map.shape) == 3:
            freqs, comp_p, N_pix = sphere_map.shape
            map_new = np.ones((freqs, comp_p, 5 * self.nside+128*2, 5 * self.nside+128*2))
            for j in range(comp_p):
                map_new[:, j, :, :] = sp.sphere2piecePlane_squa_pad_mult(sphere_map=sphere_map[:, j, :], nside=self.nside)
        else:
            map_new = sp.sphere2piecePlane_squa_pad_mult(sphere_map=sphere_map, nside=self.nside)
        return map_new

    def flat_map_from_sphere_block_map(self, sphere_map):
        if len(sphere_map.shape) == 3:
            freqs, comp_p, N_pix = sphere_map.shape
            map_new = np.ones((freqs, comp_p, 4 * self.nside, 3 * self.nside))
            map_new_block = np.ones((freqs, comp_p, self.nside, self.nside))

            for j in range(comp_p):
                map_new[:, j, :, :] = sp.sphere2piecePlane_mult(sphere_map=sphere_map[:, j, :], nside=self.nside)
                map_new_block[:, j, :, :] = sp.piecePlanes2blocks_mult(piece_maps=map_new[:, j, :, :], nside=self.nside,
                                                                      block_n=self.block)
        else:
            map_new = sp.sphere2piecePlane_mult(sphere_map=sphere_map, nside=self.nside)
            map_new_block = sp.piecePlanes2blocks_mult(piece_maps=map_new[:, :], nside=self.nside,
                                                               block_n=self.block)
        return map_new_block



    def get_flat_map_from_sphere_fullsky(self, Nsample):
        pbar = tqdm(total=len(Nsample), miniters=1)
        pbar.set_description('Converting spherical full-sky map to planar map')
        for idx in Nsample:
            if self.dataset_type == 'traing_set':
                if self.using_ilc_cmbmap:
                    cmb_obs = np.load(self.dir_cmb_ilc_tra + 'cmb' + str(idx) + '.npy')
                else:
                    cmb_obs = np.load(self.dir_cmb_obs_tra + 'cmb' + str(idx) + '.npy')
                total_obs = np.load(self.dir_total_obs_tra + 'total' + str(idx) + '.npy')
                if self.padding:
                    cmb_obs_flat = self.flat_map_from_sphere_fullsky_padding(cmb_obs)
                    total_obs_flat = self.flat_map_from_sphere_fullsky_padding(total_obs)
                else:
                    cmb_obs_flat = self.flat_map_from_sphere_fullsky(cmb_obs)
                    total_obs_flat = self.flat_map_from_sphere_fullsky(total_obs)
                if self.using_ilc_cmbmap:
                    self._creat_file(self.dir_cmb_ilc_flat_tra)
                    np.save(self.dir_cmb_ilc_flat_tra + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                else:
                    self._creat_file(self.dir_cmb_obs_flat_tra)
                    np.save(self.dir_cmb_obs_flat_tra + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                self._creat_file(self.dir_total_obs_flat_tra)
                np.save(self.dir_total_obs_flat_tra + 'total' + str(idx) + '.npy', (total_obs_flat).astype(np.float32))

            elif self.dataset_type == 'validation_set':
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_obs_1 = np.load(self.dir_cmb_ilc_val_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_ilc_val_2 + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs_1 = np.load(self.dir_cmb_obs_val_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_obs_val_2 + 'cmb' + str(idx) + '.npy')
                    total_obs_1 = np.load(self.dir_total_obs_val_1 + 'total' + str(idx) + '.npy')
                    total_obs_2 = np.load(self.dir_total_obs_val_2 + 'total' + str(idx) + '.npy')
                    if self.padding:
                        cmb_obs_flat_1 = self.flat_map_from_sphere_fullsky_padding(cmb_obs_1)
                        cmb_obs_flat_2 = self.flat_map_from_sphere_fullsky_padding(cmb_obs_2)
                        total_obs_flat_1 = self.flat_map_from_sphere_fullsky_padding(total_obs_1)
                        total_obs_flat_2 = self.flat_map_from_sphere_fullsky_padding(total_obs_2)
                    else:
                        cmb_obs_flat_1 = self.flat_map_from_sphere_fullsky(cmb_obs_1)
                        cmb_obs_flat_2 = self.flat_map_from_sphere_fullsky(cmb_obs_2)
                        total_obs_flat_1 = self.flat_map_from_sphere_fullsky(total_obs_1)
                        total_obs_flat_2 = self.flat_map_from_sphere_fullsky(total_obs_2)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_val_1)
                        self._creat_file(self.dir_cmb_ilc_flat_val_2)
                        np.save(self.dir_cmb_ilc_flat_val_1 + 'cmb' + str(idx) + '.npy', (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_ilc_flat_val_2 + 'cmb' + str(idx) + '.npy',
                            (cmb_obs_flat_2).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_val_1)
                        self._creat_file(self.dir_cmb_obs_flat_val_2)
                        np.save(self.dir_cmb_obs_flat_val_1 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_obs_flat_val_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_val_1)
                    self._creat_file(self.dir_total_obs_flat_val_2)
                    np.save(self.dir_total_obs_flat_val_1 + 'total' + str(idx) + '.npy', (total_obs_flat_1).astype(np.float32))
                    np.save(self.dir_total_obs_flat_val_2 + 'total' + str(idx) + '.npy',
                            (total_obs_flat_2).astype(np.float32))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_obs = np.load(self.dir_cmb_ilc_val + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs = np.load(self.dir_cmb_obs_val + 'cmb' + str(idx) + '.npy')
                    total_obs = np.load(self.dir_total_obs_val + 'total' + str(idx) + '.npy')
                    if self.padding:
                        cmb_obs_flat = self.flat_map_from_sphere_fullsky_padding(cmb_obs)
                        total_obs_flat = self.flat_map_from_sphere_fullsky_padding(total_obs)
                    else:
                        cmb_obs_flat = self.flat_map_from_sphere_fullsky(cmb_obs)
                        total_obs_flat = self.flat_map_from_sphere_fullsky(total_obs)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_val)
                        np.save(self.dir_cmb_ilc_flat_val + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_val)
                        np.save(self.dir_cmb_obs_flat_val + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_val)
                    np.save(self.dir_total_obs_flat_val + 'total' + str(idx) + '.npy',
                            (total_obs_flat).astype(np.float32))

            elif self.dataset_type == 'testing_set':
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_obs_1 = np.load(self.dir_cmb_ilc_tes_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_ilc_tes_2 + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs_1 = np.load(self.dir_cmb_obs_tes_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_obs_tes_2 + 'cmb' + str(idx) + '.npy')
                    total_obs_1 = np.load(self.dir_total_obs_tes_1 + 'total' + str(idx) + '.npy')
                    total_obs_2 = np.load(self.dir_total_obs_tes_2 + 'total' + str(idx) + '.npy')
                    if self.padding:
                        cmb_obs_flat_1 = self.flat_map_from_sphere_fullsky_padding(cmb_obs_1)
                        cmb_obs_flat_2 = self.flat_map_from_sphere_fullsky_padding(cmb_obs_2)
                        total_obs_flat_1 = self.flat_map_from_sphere_fullsky_padding(total_obs_1)
                        total_obs_flat_2 = self.flat_map_from_sphere_fullsky_padding(total_obs_2)
                    else:
                        cmb_obs_flat_1 = self.flat_map_from_sphere_fullsky(cmb_obs_1)
                        cmb_obs_flat_2 = self.flat_map_from_sphere_fullsky(cmb_obs_2)
                        total_obs_flat_1 = self.flat_map_from_sphere_fullsky(total_obs_1)
                        total_obs_flat_2 = self.flat_map_from_sphere_fullsky(total_obs_2)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_tes_1)
                        self._creat_file(self.dir_cmb_ilc_flat_tes_2)
                        np.save(self.dir_cmb_ilc_flat_tes_1 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_ilc_flat_tes_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_tes_1)
                        self._creat_file(self.dir_cmb_obs_flat_tes_2)
                        np.save(self.dir_cmb_obs_flat_tes_1 + 'cmb' + str(idx) + '.npy', (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_obs_flat_tes_2 + 'cmb' + str(idx) + '.npy',
                            (cmb_obs_flat_2).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_tes_1)
                    self._creat_file(self.dir_total_obs_flat_tes_2)
                    np.save(self.dir_total_obs_flat_tes_1 + 'total' + str(idx) + '.npy', (total_obs_flat_1).astype(np.float32))
                    np.save(self.dir_total_obs_flat_tes_2 + 'total' + str(idx) + '.npy',
                            (total_obs_flat_2).astype(np.float32))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_obs = np.load(self.dir_cmb_ilc_tes + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs = np.load(self.dir_cmb_obs_tes + 'cmb' + str(idx) + '.npy')
                    total_obs = np.load(self.dir_total_obs_tes + 'total' + str(idx) + '.npy')
                    if self.padding:
                        cmb_obs_flat = self.flat_map_from_sphere_fullsky_padding(cmb_obs)
                        total_obs_flat = self.flat_map_from_sphere_fullsky_padding(total_obs)
                    else:
                        cmb_obs_flat = self.flat_map_from_sphere_fullsky(cmb_obs)
                        total_obs_flat = self.flat_map_from_sphere_fullsky(total_obs)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_tes)
                        np.save(self.dir_cmb_ilc_flat_tes + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_tes)
                        np.save(self.dir_cmb_obs_flat_tes + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_tes)
                    np.save(self.dir_total_obs_flat_tes + 'total' + str(idx) + '.npy',
                            (total_obs_flat).astype(np.float32))
            else:
                print('An error occurred: Please set the dataset_type correctly. dataset_type = "traing_set" or "validation_set" or "testing_set"')
            pbar.update(1)


    def get_flat_map_from_sphere_blocksky(self, Nsample):
        pbar = tqdm(total=len(Nsample), miniters=1)
        pbar.set_description('Converting spherical partial-sky map to planar map')
        for idx in Nsample:
            if self.dataset_type == 'traing_set':
                if self.using_ilc_cmbmap:
                    cmb_obs = np.load(self.dir_cmb_ilc_tra + 'cmb' + str(idx) + '.npy')
                else:
                    cmb_obs = np.load(self.dir_cmb_obs_tra + 'cmb' + str(idx) + '.npy')
                total_obs = np.load(self.dir_total_obs_tra + 'total' + str(idx) + '.npy')
                cmb_obs_flat = self.flat_map_from_sphere_block_map(cmb_obs)
                total_obs_flat = self.flat_map_from_sphere_block_map(total_obs)
                if self.using_ilc_cmbmap:
                    self._creat_file(self.dir_cmb_ilc_flat_block_tra)
                    np.save(self.dir_cmb_ilc_flat_block_tra + 'cmb' + str(idx) + '.npy',
                            (cmb_obs_flat).astype(np.float32))
                else:
                    self._creat_file(self.dir_cmb_obs_flat_block_tra)
                    np.save(self.dir_cmb_obs_flat_block_tra + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                self._creat_file(self.dir_total_obs_flat_block_tra)
                np.save(self.dir_total_obs_flat_block_tra + 'total' + str(idx) + '.npy', (total_obs_flat).astype(np.float32))

            elif self.dataset_type == 'validation_set':
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_obs_1 = np.load(self.dir_cmb_ilc_val_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_ilc_val_2 + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs_1 = np.load(self.dir_cmb_obs_val_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_obs_val_2 + 'cmb' + str(idx) + '.npy')
                    total_obs_1 = np.load(self.dir_total_obs_val_1 + 'total' + str(idx) + '.npy')
                    total_obs_2 = np.load(self.dir_total_obs_val_2 + 'total' + str(idx) + '.npy')
                    cmb_obs_flat_1 = self.flat_map_from_sphere_block_map(cmb_obs_1)
                    cmb_obs_flat_2 = self.flat_map_from_sphere_block_map(cmb_obs_2)
                    total_obs_flat_1 = self.flat_map_from_sphere_block_map(total_obs_1)
                    total_obs_flat_2 = self.flat_map_from_sphere_block_map(total_obs_2)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_block_val_1)
                        self._creat_file(self.dir_cmb_ilc_flat_block_val_2)
                        np.save(self.dir_cmb_ilc_flat_block_val_1 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_ilc_flat_block_val_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_block_val_1)
                        self._creat_file(self.dir_cmb_obs_flat_block_val_2)
                        np.save(self.dir_cmb_obs_flat_block_val_1 + 'cmb' + str(idx) + '.npy', (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_obs_flat_block_val_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_block_val_1)
                    self._creat_file(self.dir_total_obs_flat_block_val_2)
                    np.save(self.dir_total_obs_flat_block_val_1 + 'total' + str(idx) + '.npy', (total_obs_flat_1).astype(np.float32))
                    np.save(self.dir_total_obs_flat_block_val_2 + 'total' + str(idx) + '.npy',
                            (total_obs_flat_2).astype(np.float32))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_obs = np.load(self.dir_cmb_ilc_val + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs = np.load(self.dir_cmb_obs_val + 'cmb' + str(idx) + '.npy')
                    total_obs = np.load(self.dir_total_obs_val + 'total' + str(idx) + '.npy')
                    cmb_obs_flat = self.flat_map_from_sphere_block_map(cmb_obs)
                    total_obs_flat = self.flat_map_from_sphere_block_map(total_obs)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_block_val)
                        np.save(self.dir_cmb_ilc_flat_block_val + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_block_val)
                        np.save(self.dir_cmb_obs_flat_block_val + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_block_val)
                    np.save(self.dir_total_obs_flat_block_val + 'total' + str(idx) + '.npy',
                            (total_obs_flat).astype(np.float32))

            elif self.dataset_type == 'testing_set':
                if self.is_half_split_map:
                    if self.using_ilc_cmbmap:
                        cmb_obs_1 = np.load(self.dir_cmb_ilc_tes_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_ilc_tes_2 + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs_1 = np.load(self.dir_cmb_obs_tes_1 + 'cmb' + str(idx) + '.npy')
                        cmb_obs_2 = np.load(self.dir_cmb_obs_tes_2 + 'cmb' + str(idx) + '.npy')
                    total_obs_1 = np.load(self.dir_total_obs_tes_1 + 'total' + str(idx) + '.npy')
                    total_obs_2 = np.load(self.dir_total_obs_tes_2 + 'total' + str(idx) + '.npy')
                    cmb_obs_flat_1 = self.flat_map_from_sphere_block_map(cmb_obs_1)
                    cmb_obs_flat_2 = self.flat_map_from_sphere_block_map(cmb_obs_2)
                    total_obs_flat_1 = self.flat_map_from_sphere_block_map(total_obs_1)
                    total_obs_flat_2 = self.flat_map_from_sphere_block_map(total_obs_2)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_block_tes_1)
                        self._creat_file(self.dir_cmb_ilc_flat_block_tes_2)
                        np.save(self.dir_cmb_ilc_flat_block_tes_1 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_ilc_flat_block_tes_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_block_tes_1)
                        self._creat_file(self.dir_cmb_obs_flat_block_tes_2)
                        np.save(self.dir_cmb_obs_flat_block_tes_1 + 'cmb' + str(idx) + '.npy', (cmb_obs_flat_1).astype(np.float32))
                        np.save(self.dir_cmb_obs_flat_block_tes_2 + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat_2).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_block_tes_1)
                    self._creat_file(self.dir_total_obs_flat_block_tes_2)
                    np.save(self.dir_total_obs_flat_block_tes_1 + 'total' + str(idx) + '.npy', (total_obs_flat_1).astype(np.float32))
                    np.save(self.dir_total_obs_flat_block_tes_2 + 'total' + str(idx) + '.npy',
                            (total_obs_flat_2).astype(np.float32))
                else:
                    if self.using_ilc_cmbmap:
                        cmb_obs = np.load(self.dir_cmb_ilc_tes + 'cmb' + str(idx) + '.npy')
                    else:
                        cmb_obs = np.load(self.dir_cmb_obs_tes + 'cmb' + str(idx) + '.npy')
                    total_obs = np.load(self.dir_total_obs_tes + 'total' + str(idx) + '.npy')
                    cmb_obs_flat = self.flat_map_from_sphere_block_map(cmb_obs)
                    total_obs_flat = self.flat_map_from_sphere_block_map(total_obs)
                    if self.using_ilc_cmbmap:
                        self._creat_file(self.dir_cmb_ilc_flat_block_tes)
                        np.save(self.dir_cmb_ilc_flat_block_tes + 'cmb' + str(idx) + '.npy',
                                (cmb_obs_flat).astype(np.float32))
                    else:
                        self._creat_file(self.dir_cmb_obs_flat_block_tes)
                        np.save(self.dir_cmb_obs_flat_block_tes + 'cmb' + str(idx) + '.npy', (cmb_obs_flat).astype(np.float32))
                    self._creat_file(self.dir_total_obs_flat_block_tes)
                    np.save(self.dir_total_obs_flat_block_tes + 'total' + str(idx) + '.npy',
                            (total_obs_flat).astype(np.float32))
            else:
                print('An error occurred: Please set the dataset_type correctly. dataset_type = "traing_set" or "validation_set" or "testing_set"')
            pbar.update(1)

    def mult_process_get_flatmap_from_spheremap(self, N_mult, N_sample,   dataset_type = 'traing_set'):
        '''
        Multithreaded data processing
        :param N_mult: Number of threads
        :param N_train_sample: number of  full map
        :return:
        '''
        self.dataset_type = dataset_type



        N_per = N_sample // N_mult
        N_lop = N_sample // N_per
        N_rem = N_sample % N_per
        S_N = [np.arange(i * N_per, (i + 1) * N_per) for i in range(N_lop)]
        if not N_rem == 0:
            S_N_1 = [np.arange(N_lop * N_per, N_sample)]
            S_N.extend(S_N_1)

        if self.full_sky_map:
            processes = [multiprocessing.Process(target=self.get_flat_map_from_sphere_fullsky, kwargs={'Nsample': S_N[i]}) for i in
                         range(len(S_N))]
        else:
            processes = [multiprocessing.Process(target=self.get_flat_map_from_sphere_blocksky, kwargs={'Nsample': S_N[i]}) for i in
                         range(len(S_N))]
        for process in processes:
            process.start()
            process.join()


    def sphere_map_from_fla_map_fullsky(self, piece_map):

        if len(piece_map) == 4:
            freqs, comp_p, N_pix, _ = piece_map.shape
            map_new = np.zeros((freqs, comp_p, 12 * self.nside ** 2))
            for j in range(comp_p):
                map_new[:, j, :, :] = sp.piecePlane_squa2sphere_mult(piece_map[:, j, :], nside=self.nside)
        else:
            map_new = sp.piecePlane_squa2sphere_mult(piece_map, nside=self.nside)
        return map_new

    def sphere_map_from_flat_map_fullsky_padding(self, piece_map):

        if len(piece_map) == 4:
            freqs, comp_p, N_pix, _ = piece_map.shape
            map_new = np.zeros((freqs, comp_p, 12 * self.nside ** 2))
            for j in range(comp_p):
                map_new[:, j, :, :] = sp.piecePlane_squa2sphere_pad_mult(piece_map[:, j, :], nside=self.nside)
        else:
            map_new = sp.piecePlane_squa2sphere_pad_mult(piece_map, nside=self.nside)
        return map_new

    def sphere_from_flat_block_map(self, block_map, block='block_0'):

        if len(block_map) == 4:
            freqs, comp_p, N_pix, _ = block_map.shape
            map_new = np.zeros((freqs, comp_p, 12 * self.nside ** 2))
            for j in range(comp_p):
                map_new[:, j, :] = sp.blockPlane2sphere_mult(block_map[:, j, :], nside=self.nside, block_n=block)
        else:
            map_new = sp.blockPlane2sphere_mult(block_map, nside=self.nside, block_n=block)

        return map_new

    def sphere_from_flat_map(self, flat_map, is_fullsky, is_padding, block = 'block_0'):
        if is_fullsky:
            if is_padding:
                return self.sphere_map_from_flat_map_fullsky_padding(flat_map)
            else:
                return self.sphere_map_from_fla_map_fullsky(flat_map)
        else:
            return self.sphere_from_flat_block_map(flat_map, block=block)




class Load_data(object):
    def __init__(self, map_nums,component = "Q",  input_dir = "data/", target_dir=None, output_freq=4, using_ilc_cmbmap=True):
        self.map_nums = map_nums         # batch map index
        self.input_dir = input_dir     # loaction
        self.target_dir = target_dir
        self.output_freq = output_freq
        self.using_ilc_cmbmap = using_ilc_cmbmap
        if component == "T":
            self.component = 0
        elif component == "Q":
            self.component = 1
        elif component == "U":
            self.component = 2
        else:
            print("Error: Please set component correctly. component = 'T', 'Q' or 'U' ")

    def data(self):
        input = np.load(self.input_dir + "total" + str(self.map_nums[0]) + ".npy")
        if input.shape[1] == 2:
            self.component = self.component-1
        input = input[:,self.component,:]
        input = input[None,:,:]
        input = torch.from_numpy(input.astype(np.float32))

        target = np.load(self.target_dir + "cmb" + str(self.map_nums[0]) + ".npy")
        if len(target.shape) == 4:
            target = target[self.output_freq, self.component, :]
        elif len(target.shape) == 3:
            target = target[self.component, :]
        target = target[None, None, :]
        target = torch.from_numpy(target.astype(np.float32))


        if len(self.map_nums) > 1:
            for i in self.map_nums[1:].tolist():
                fits_filename_total = self.input_dir + 'total' + str(i) + ".npy"
                fits_filename_cmb = self.target_dir + "cmb" + str(i) + ".npy"

                input_i = np.load(fits_filename_total)
                input_i = input_i[:, self.component, :]
                input_i = input_i[None, :, :]
                input_i = torch.from_numpy(input_i.astype(np.float32))

                target_i = np.load(fits_filename_cmb)
                if len(target_i.shape) == 4:
                    target_i = target_i[self.output_freq, self.component, :]
                elif len(target_i.shape) == 3:
                    target_i = target_i[self.component, :]
                target_i = target_i[None, None, :]
                target_i = torch.from_numpy(target_i.astype(np.float32))
                input = torch.cat((input, input_i), 0)
                target = torch.cat((target, target_i), 0)

        return input, target


