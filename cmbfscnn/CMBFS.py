
# -*- coding: utf-8 -*-


import numpy as np
from . import CMBFS_mode as Cmo
from . import generate_data as gd
from . import Results as Re

class CMBFSCNN(object):
    def __init__(self, config):
        self.config = config
        self._get_config

    @property
    def _get_config(self):
        for key, value in self.config.items():
            setattr(self, key, value)

        self.data_index = [np.arange(self.N_sky_maps[0]), np.arange(self.N_sky_maps[1], np.sum(self.N_sky_maps[0:2])),
                           np.arange(np.sum(self.N_sky_maps[0:2]), np.sum(self.N_sky_maps))]
        self.noise_index = [np.arange(i) for i in self.N_noise_maps]


    def data_simulation(self):

        Sim_data = gd.Simulator_data(freqs=self.freqs, beams=self.beams, noise_level=self.sens,
                                  is_half_split_map=self.is_half_split_map, using_ilc_cmbmap=self.using_ilc_cmbmap,
                                  Nside=self.nside, config_random=self.data_config_random, save_dir=self.save_data_dir,
                                  is_polarization_data=self.is_polarization_data)
        Sim_data.simulator_sky_map(np.sum(self.N_sky_maps))

        for data_type, N_noise in zip(self.dataset_type, self.N_noise_maps):
            Sim_data.simulator_noise_map(N_noise_map = N_noise, dataset_type=data_type)

        for k, data_type in enumerate(self.dataset_type):

            Sim_data.get_observed_map(index_sky_map = self.data_index[k], index_noise_map = self.noise_index[k],
                                      dataset_type = data_type)

    def cal_ilc_noise(self):
        Sim_data = gd.Simulator_data(freqs=self.freqs, beams=self.beams, noise_level=self.sens,
                                  is_half_split_map=self.is_half_split_map, using_ilc_cmbmap=self.using_ilc_cmbmap,
                                  Nside=self.nside, config_random=self.data_config_random, save_dir=self.save_data_dir,
                                  is_polarization_data=self.is_polarization_data)

        if self.using_ilc_cmbmap:
            for k, data_type in enumerate(self.dataset_type):
                Sim_data.mult_process_get_ilc_noise(N_sample = self.N_sky_maps[k], N_mult = self.ILC_N_threads, out_freq = self.output_freq, mask = self.ilc_mask,
                                                    dataset_type=data_type)
            for k, data_type in enumerate(self.dataset_type):
                if data_type == 'traing_set':
                    Sim_data.get_data_CMB_ilcnoise(index_sky_map=self.data_index[k],
                                                   index_ilcnoise_map=self.data_index[k],
                                                                                dataset_type=data_type,
                                                                                out_beam=self.output_beam)
                else:
                    Sim_data.get_data_CMB_ilcnoise(index_sky_map = self.data_index[k],
                                                   index_ilcnoise_map = self.noise_index[k],
                                                   dataset_type = data_type, out_beam = self.output_beam)


    def data_preprocessing(self):
        Data_prep = gd.Data_preprocessing(save_dir = self.save_data_dir, padding = self.padding, is_half_split_map = self.is_half_split_map,
                                       nside = self.nside, block_number = self.block_n, full_sky_map = self.is_fullsky,using_ilc_cmbmap=self.using_ilc_cmbmap)
        for k, data_type in enumerate(self.dataset_type):
            Data_prep.mult_process_get_flatmap_from_spheremap(N_mult = self.N_threads_preprocessing, N_sample = self.N_sky_maps[k], dataset_type = data_type)


    def _training_cnn(self, component = 'Q', using_loss_fft = True, repeat_n = 3):
        FSM_Qmap = Cmo.Foreground_subtraction_model(data_dir = self.save_data_dir, freqs = self.freqs, output_freq = self.output_freq,
                                                component = component, full_sky_map = self.is_fullsky,
                                                is_half_split_map = self.is_half_split_map, using_ilc_cmbmap = self.using_ilc_cmbmap,
                                                result_dir = self.save_result_dir)
        FSM_Qmap.net_train(batch_size = self.batch_size, learning_rate = self.learning_rate, num_train = self.N_sky_maps[0],
                           num_validation = self.N_sky_maps[1], device_ids = self.device_ids,
                           iteration = self.iteration,
                           CNN_model = self.CNN_model, using_loss_fft = using_loss_fft, repeat_n = repeat_n)
        FSM_Qmap.plot_train_records()  # Plot Training History

    def training_cnn(self):
        for st in self.component:
            self._training_cnn(st)


    def get_predicted_maps(self):
        RA = Cmo.Result_analysis(data_dir = self.save_data_dir, full_sky_map = self.is_fullsky, map_block = self.block_n, padding = self.padding,
                             using_ilc_cmbmap = self.using_ilc_cmbmap, nside = self.nside,
                             is_half_split_map = self.is_half_split_map, result_dir = self.save_result_dir, freqs = self.freqs,
                             output_freq = self.output_freq)
        for st in self.component:
            RA.prediction(num_testset = self.N_sky_maps[2], is_return=False, comp = st)

        RA.get_true_CMB_map(index_sky_map = self.data_index[2])

    def calculate_power_spectra(self, nlb=5, Dl = True):
        CPS = Cmo.Calculate_power_spectra(result_dir = self.save_result_dir, is_half_split_map = self.is_half_split_map,
                                      beams_arcmin = self.beams, output_beam_arcmin = self.output_beam, N_sample = self.N_sky_maps[2],
                                      nside = self.nside, component = self.component)

        if 'T' in self.component:
            CPS.get_true_CMB_PS(nlb=nlb, Dl=Dl, aposize=None, EB_power=False)  # get true CMB TT and TT power spectra
            CPS.cal_cmb_T_PS(nlb=nlb, Dl=Dl, aposize=None, EB_power=False)  # get output CMB TT and TT power spectra
            if self.is_half_split_map:
                CPS.cal_cmb_T_cross_PS(nlb=nlb, Dl=Dl, aposize=None,
                                       EB_power=False)  # get output CMB TT and TT power spectra after denoising step
        if 'Q'in self.component and 'U'in self.component:
            CPS.get_true_CMB_PS(nlb = nlb, Dl = Dl, aposize = None, EB_power = False)  # get true CMB QQ and UU power spectra
            CPS.get_true_CMB_PS(nlb = nlb, Dl = Dl, aposize = None, EB_power = True)  # get true CMB EE and BB power spectra
            CPS.cal_cmb_Q_or_U_PS(nlb = nlb, Dl = Dl, aposize = None, EB_power=False)  # get output CMB QQ and UU power spectra
            CPS.cal_cmb_Q_or_U_cross_PS(nlb = nlb, Dl = Dl, aposize = None,
                                        EB_power = False)  # get output CMB QQ and UU power spectra after denoising step
            CPS.cal_cmb_E_B_PS(nlb = nlb, Dl = Dl, aposize = None, EB_power = True)  # get output CMB EE and BB power spectra
            CPS.cal_cmb_E_B_cross_PS(nlb = nlb, Dl = Dl, aposize = None,
                                     EB_power = True)  # get output CMB EE and BB power spectra after denoising step

    def _plot_results(self):
        PR = Re.Plot_results(result_dir = self.save_result_dir, is_half_split_map = self.is_half_split_map)
        PR.plot_predicted_sphere_map()
        PR.plot_predicted_flat_map()
        PR.plot_recovered_CMB_QU_PS()
        PR.plot_recovered_CMB_EB_PS()


    def run_CMBFSCNN(self):
        self.data_simulation()
        if self.using_ilc_cmbmap:
            self.cal_ilc_noise()
        self.data_preprocessing()
        self.training_cnn()
        self.get_predicted_maps()
        self.calculate_power_spectra()
        self._plot_results()






















