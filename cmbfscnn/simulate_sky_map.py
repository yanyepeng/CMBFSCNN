# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import healpy as hp
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import pysm
from pysm.nominal import models
from pysm.common import convert_units

from . import get_power_sperctra as ps

def c2_mode(nside):
    return [{
        'model' : 'taylens',
        'nside' : nside,
        'cmb_seed' : 1111,
        'delens': False,
        'output_unlens': False
        }]


def c2_unlens_mode(nside):
    return [{
        'model' : 'taylens',
        'nside' : nside,
        'cmb_seed' : 1111,
        'delens': False,
        'output_unlens':True
        }]


def get_specificRandn(n, mu, sigma, range_min, range_max):
    randn = np.random.randn(n) * sigma + mu
    choose_1 = np.where(randn>=range_min)
    randn = randn[choose_1]
    choose_2 = np.where(randn<=range_max)
    randn = randn[choose_2]
    return randn


class Get_data(object):
    def __init__(self, Nside,  config_random = {}, freqs=None,  using_beam = False,
                  beam = None, out_unit = None):
        self.Nside = Nside
        self.freqs = freqs
        self.using_beam = using_beam
        self.beam = beam
        self.out_unit = out_unit # the unit of output, 'K_CMB', 'Jysr', 'uK_RJ';
        # Default unit of signals is 'uK_RJ'; Default unit of noise may be 'K_CMB'
        self.config_random = config_random

    def data(self):
        # random can be 'fixed', fix cosmological paramaters

        cmb_specs = ps.ParametersSampling(random=self.config_random['Random_types_of_cosmological_parameters'], spectra_type='unlensed_scalar')
        s1 = models("s1", self.Nside)
        d1 = models("d1", self.Nside)
        a2 = models("a2", self.Nside)
        c2 = c2_mode(self.Nside)
        c2[0]['cmb_specs'] = cmb_specs
        c1_seed = self.config_random['cmb_seed']
        c2[0]['cmb_seed'] = c1_seed

        c2_unlens = c2_unlens_mode(self.Nside)
        cmb_spe = cmb_specs.copy()
        c2_unlens[0]['cmb_specs'] = cmb_spe
        c2_unlens[0]['cmb_seed'] = c1_seed

        # random of syn parameters
        # np.random.seed(int(seed+100**1))
        s1_seed = self.config_random['syn_seed']
        np.random.seed(int(s1_seed))
        if 'syn_spectralindex_random' not in self.config_random:
            s1_index_std = 0
        else:
            s1_index_std, syn_index_ran_class = self.config_random['syn_spectralindex_random']
        if s1_index_std == 0:
            print('Note! syn_spectralindex are not random')
        else:
            if syn_index_ran_class == 'one' :
                s1[0]['spectral_index'] = s1[0]['spectral_index'] + \
                                          get_specificRandn(50, 0, s1_index_std, -2 * s1_index_std, 2 * s1_index_std)[
                                              0] * (s1[0]['spectral_index'])
            elif syn_index_ran_class == 'multi' :
                pixel_n = len(s1[0]['spectral_index'])
                s1[0]['spectral_index'] = s1[0]['spectral_index'] + \
                                          get_specificRandn(pixel_n * 2, 0, s1_index_std,
                                                                                      -2 * s1_index_std,
                                                                                      2 * s1_index_std)[:pixel_n] * (
                                                      s1[0]['spectral_index'])
            else:
                print('Note! syn_spectralindex config error')

        if 'syn_amplitude_random' not in self.config_random:
            s1_A_std = 0
        else:
            s1_A_std, syn_A_ran_class = self.config_random['syn_amplitude_random']
        if s1_A_std == 0:
            print('Note! syn_amplitude are not random')
        else:
            if syn_A_ran_class == 'one':
                s1[0]['A_I'] = s1[0]['A_I'] + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * \
                               s1[0]['A_I']
                s1[0]['A_Q'] = s1[0]['A_Q'] + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * \
                               s1[0]['A_Q']
                s1[0]['A_U'] = s1[0]['A_U'] + get_specificRandn(50, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[0] * \
                               s1[0]['A_U']
            elif syn_A_ran_class == 'multi':
                pixel_n = len(s1[0]['A_I'])
                s1[0]['A_I'] = s1[0]['A_I'] + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[
                                              :pixel_n] * s1[0]['A_I']
                s1[0]['A_Q'] = s1[0]['A_Q'] + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[
                                              :pixel_n] * s1[0]['A_Q']
                s1[0]['A_U'] = s1[0]['A_U'] + get_specificRandn(pixel_n*2, 0, s1_A_std, -2 * s1_A_std, 2 * s1_A_std)[
                                              :pixel_n] * s1[0]['A_U']
            else:
                print('Note! syn_amplitude_random config error')

        # random of dust parameters
        # np.random.seed(int(seed + 100 ** 2))
        d1_seed = self.config_random['dust_seed']
        np.random.seed(int(d1_seed))
        if 'dust_amplitude_random' not in self.config_random:
            d1_A_std = 0
        else:
            d1_A_std, dust_A_ran_class = self.config_random['dust_amplitude_random']
        if d1_A_std == 0 :
            print('Note! dust_amplitude are not random')
        else:
            if dust_A_ran_class == 'one':
                d1[0]['A_I'] = d1[0]['A_I'] + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * \
                               d1[0]['A_I']
                d1[0]['A_Q'] = d1[0]['A_Q'] + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * \
                               d1[0]['A_Q']
                d1[0]['A_U'] = d1[0]['A_U'] + get_specificRandn(50, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[0] * \
                               d1[0]['A_U']
            elif dust_A_ran_class == 'multi':
                pixel_n = len(d1[0]['A_I'])
                d1[0]['A_I'] = d1[0]['A_I'] + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * \
                               d1[0]['A_I']
                d1[0]['A_Q'] = d1[0]['A_Q'] + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * \
                               d1[0]['A_Q']
                d1[0]['A_U'] = d1[0]['A_U'] + get_specificRandn(pixel_n*2, 0, d1_A_std, -2 * d1_A_std, 2 * d1_A_std)[:pixel_n] * \
                               d1[0]['A_U']
            else:
                print('Note! dust_amplitude config error')

        if 'dust_spectralindex_random' not in self.config_random:
            d1_index_std = 0
        else:
            d1_index_std, dust_index_ran_class = self.config_random['dust_spectralindex_random']
        if d1_index_std == 0:
            print('Note! dust_spectralindex are not random')
        else:
            if dust_index_ran_class == 'one' :
                mss = d1[0]['spectral_index']
                d1[0]['spectral_index'] = mss + \
                                          get_specificRandn(50, 0, d1_index_std, -2 * d1_index_std, 2 * d1_index_std)[
                                              0] * (mss - 2.)
            elif dust_index_ran_class == 'multi' :
                pixel_n = len(s1[0]['spectral_index'])
                mss = d1[0]['spectral_index']
                d1[0]['spectral_index'] = mss + \
                                          get_specificRandn(pixel_n*2, 0, d1_index_std, -2 * d1_index_std, 2 * d1_index_std)[
                                              :pixel_n] * (mss - 2.)
            else:
                print('Note! syn_spectralindex config error')

        if 'dust_temp_random' not in self.config_random:
            d1_temp_std = 0
        else:
            d1_temp_std, dust_temp_ran_class = self.config_random['dust_temp_random']
        if d1_temp_std == 0:
            print('Note! dust_temp are not random')
        else:
            if dust_temp_ran_class == 'one':
                d1[0]['temp'] = d1[0]['temp'] + get_specificRandn(50, 0, d1_temp_std, -2 * d1_temp_std, 2 * d1_temp_std)[0] * \
                               d1[0]['temp']
            elif dust_temp_ran_class == 'multi':
                pixel_n = len(d1[0]['temp'])
                d1[0]['temp'] = d1[0]['temp'] + get_specificRandn(pixel_n*2, 0, d1_temp_std, -2 * d1_temp_std, 2 * d1_temp_std)[
                                              :pixel_n] * \
                               d1[0]['temp']
            else:
                print('Note! syn_temp config error')

        a1_seed = self.config_random['ame_seed']
        np.random.seed(int(a1_seed))
        if 'ame_amplitude_random' not in self.config_random:
            a1_A_std = 0
        else:
            a1_A_std, ame_A_ran_class = self.config_random['ame_amplitude_random']
        if a1_A_std == 0:
            print('Note! ame_amplitude are not random')
        else:
            if ame_A_ran_class == 'one':
                a2[0]['A_I'] = a2[0]['A_I'] + get_specificRandn(50, 0, a1_A_std, -2 * a1_A_std, 2 * a1_A_std)[0] * \
                               a2[0]['A_I']
            elif ame_A_ran_class == 'multi':
                pixel_n = len(a2[0]['A_I'])
                a2[0]['A_I'] = a2[0]['A_I'] + get_specificRandn(pixel_n*2, 0, a1_A_std, -2 * a1_A_std, 2 * a1_A_std)[:pixel_n] * \
                               a2[0]['A_I']
            else:
                print('Note! ame_A config error')

        sky_config = {
            'synchrotron': s1,
            'dust': d1,
            'cmb': c2,
            'ame': a2,
            # 'freefree': f1
        }
        sky = pysm.Sky(sky_config)  #

        total = sky.signal()(self.freqs)
        total = total.astype(np.float32)

        cmb = sky.cmb(self.freqs)
        cmb = cmb.astype(np.float32)
        if not self.out_unit == None:
            Uc_signal = np.array(convert_units("uK_RJ", self.out_unit, self.freqs))
            if not len(self.freqs)>1:  # one frequence
                cmb = cmb * Uc_signal[:, None, None]
                total = total * Uc_signal[:, None, None]
            else:
                cmb = cmb * Uc_signal[:, None, None]
                total = total * Uc_signal[:, None, None]

        cmb = self.data_proce_beam(cmb,using_beam_1=self.using_beam, beam_1=self.beam)
        total = self.data_proce_beam(total, using_beam_1=self.using_beam, beam_1=self.beam)
        return cmb, total

    def data_proce_beam(self, map_da,using_beam_1=False, beam_1=None):
        if using_beam_1:
            beam = beam_1
            map_n = np.array(
                [hp.smoothing(m, fwhm=np.pi / 180. * b / 60., verbose=False) for (m, b) in zip(map_da, beam)])
        else:
            map_n = map_da
            # dd = map_new
        return map_n

    def noiser(self, Sens, is_half_split_map = True):
        """Calculate white noise maps for given sensitivities.  Returns noise, and noise maps at the given nside in (T, Q, U). Input
        sensitivities are expected to be in uK_CMB amin for the rest of
        PySM.

        :param is_half_split_map: If it is an half-split map, the noise level will increase by sqrt(2) times

        """
        # solid angle per pixel in amin2
        npix = hp.nside2npix(self.Nside)
        # solid angle per pixel in amin2, Note!!!!!
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.Nside)) * (180. * 60. / np.pi) ** 2
        """sigma_pix_I/P is std of noise per pixel. It is an array of length
        equal to the number of input maps."""
        if is_half_split_map:
            sigma_pix_I = np.sqrt(Sens ** 2 / pix_amin2)*np.sqrt(2)
        else:
            sigma_pix_I = np.sqrt(Sens ** 2 / pix_amin2)
        noise = np.random.randn(len(Sens), 3,npix)
        noise *= sigma_pix_I[:, None,None]
        if not self.out_unit ==None:
            Uc_noise = np.array(convert_units("uK_CMB", self.out_unit, self.freqs))
            noise = noise * Uc_noise[:, None, None]
        return noise.astype(np.float32)






