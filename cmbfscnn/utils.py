
import pickle
import numpy as np
import healpy as hp
from . import namaster as na
from. import spherical as sp
from tqdm import tqdm


def save_pkl(obj, pkl_name):
    with open(pkl_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(pkl_name):
    with open(pkl_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def MAD(output_data, target):
    MAD = np.mean(abs(output_data - target))
    return MAD

# def write_yaml(data, dir=''):
#     file = open(dir, 'w', encoding='utf-8')  # w:
#     # yaml.dump(data, file,allow_unicode=True)
#     yaml.safe_dump(data, file, allow_unicode=True)
#     file.close()
#
# def read_yaml(dir = ''):
#     f = open(dir, 'r',encoding='utf-8')
#     conf_str = f.read()
#     f.close()
#     return yaml.load(conf_str, Loader=yaml.FullLoader)

def get_mask(partial_map):
    mask = np.where(partial_map==0,0,1)
    return mask

def get_Bl(fwhm,nside=512):
    lmax = 3*nside - 1
    gauss_beam = hp.gauss_beam(fwhm * np.pi / 10800., lmax=lmax)
    return gauss_beam


def get_full_map(Q_map, block_n=0):
    if not block_n==None:
        Q_map = sp.Block2Full(Q_map, block_n, base_map=None).full()
    return Q_map

def get_mask(partial_map):
    mask = np.where(abs(partial_map) > 1e-6, 1, 0)
    return mask

def get_mask_for_block(block = 'block_0', nside=512):
    partial_map = np.ones(12*nside**2)
    map_new = sp.sphere2piecePlane_mult(sphere_map=partial_map, nside=nside)
    map_new_block = sp.piecePlanes2blocks_mult(piece_maps=map_new, nside=nside,
                                                           block_n=block)
    map_new = sp.blockPlane2sphere_mult(map_new_block, nside=nside, block_n=block)
    mask = np.where(abs(map_new) > 1e-6, 1, 0)
    return mask

class Get_power(object):
    def __init__(self, data_Q1_dir=None, data_Q2_dir = None,  data_U1_dir=None, data_U2_dir = None, block_n=0, nside=512,mask=None,
                 beam_file=None,nlb=None,Dl=None,aposize=None):
        self.data_Q1_dir = data_Q1_dir
        self.data_Q2_dir = data_Q2_dir
        self.data_U1_dir = data_U1_dir
        self.data_U2_dir = data_U2_dir
        self.block_n = block_n
        self.nside = nside
        if data_Q2_dir==None:
            self.data_Q2_dir =self.data_Q1_dir
        if data_U2_dir==None:
            self.data_U2_dir =self.data_U1_dir
        self.mask = mask
        self.beam_file = beam_file
        self.nlb = nlb
        self.Dl = Dl
        self.aposize = aposize


    def cal_Cl(self,map1, map2):
        denoise_power_spin2 = na.Calculate_power_spectrum(map1, map2, self.mask, aposize=self.aposize,
                                                              Bl=self.beam_file, nside=self.nside, nlb=self.nlb,
                                                              Dl=self.Dl)
        if map1.ndim==2:
            ell, powe_de = denoise_power_spin2.get_power_spectra_for_spin2()
            EE, BB = powe_de[0, :], powe_de[3, :]
            return ell, EE, BB
        elif map1.ndim==1:
            ell, powe = denoise_power_spin2.get_power_spectra_for_spin1()
            return ell, powe
        else:
            print('ERROR: shape of maps have error')

    def map_QU(self, Q_map, U_map=None):
        if type(U_map) == np.ndarray:
            if not self.block_n == None:
                Q_map = get_full_map(Q_map)
                U_map = get_full_map(U_map)
            else:
                Q_map = sp.piecePlane2sphere(Q_map, nside=self.nside)
                U_map = sp.piecePlane2sphere(U_map, nside=self.nside)
            full_map = np.zeros((2, len(Q_map)))
            full_map[0, :] = Q_map
            full_map[1, :] = U_map
            return full_map
        else:
            if not self.block_n == None:
                Q_map = get_full_map(Q_map)
            else:
                Q_map = sp.piecePlane2sphere(Q_map, nside=self.nside)
            return Q_map

    def get_N_power(self, N_sample, EB_power = False):
        if EB_power:
            Q1 = np.load(self.data_Q1_dir)
            Q2 = np.load(self.data_Q2_dir)
            U1 = np.load(self.data_U1_dir)
            U2 = np.load(self.data_U2_dir)
        else:
            Q1 = np.load(self.data_Q1_dir)
            Q2 = np.load(self.data_Q2_dir)

        if self.mask is None:
            self.mask = get_mask(Q1[0,:])
        pbar = tqdm(total=len(N_sample))
        for ite, nn in enumerate(N_sample):


            if EB_power:
                full_map1 = self.map_QU(Q1[nn,:], U1[nn,:])
                full_map2 = self.map_QU(Q2[nn,:], U2[nn,:])
                ell, EE, BB = self.cal_Cl(full_map1, full_map2)
                spectra = np.vstack((ell, EE, BB))
                if ite == 0:
                    spectra_N = spectra[None, :]
                else:
                    spectra_N = np.vstack((spectra_N, spectra[None, :]))
            else:
                Q1_i = self.map_QU(Q1[nn,:])
                Q2_i = self.map_QU(Q2[nn,:])
                ell, pow = self.cal_Cl(Q1_i, Q2_i)
                spectra = np.vstack((ell, pow))
                if ite == 0:
                    spectra_N = spectra[None, :]
                else:
                    spectra_N = np.vstack((spectra_N, spectra[None, :]))
            pbar.set_description('Calculating power spectra')
            pbar.update(1)

        return spectra_N

    def get_N_power_from_spheremap(self, N_sample, EB_power = False):
        if EB_power:
            Q1 = np.load(self.data_Q1_dir)
            Q2 = np.load(self.data_Q2_dir)
            U1 = np.load(self.data_U1_dir)
            U2 = np.load(self.data_U2_dir)
        else:
            Q1 = np.load(self.data_Q1_dir)
            Q2 = np.load(self.data_Q2_dir)

        if self.mask is None:
            self.mask = get_mask(Q1[0,:])
        pbar = tqdm(total=len(N_sample))
        for ite, nn in enumerate(N_sample):

            if EB_power:

                full_map1 = np.zeros((2, 12*self.nside**2))
                full_map1[0, :] = Q1[nn,:]
                full_map1[1, :] = U1[nn,:]
                full_map2 = np.zeros((2, 12*self.nside**2))
                full_map2[0, :] = Q2[nn, :]
                full_map2[1, :] = U2[nn, :]


                ell, EE, BB = self.cal_Cl(full_map1, full_map2)
                spectra = np.vstack((ell, EE, BB))
                if ite == 0:
                    spectra_N = spectra[None, :]
                else:
                    spectra_N = np.vstack((spectra_N, spectra[None, :]))
            else:
                Q1_i = Q1[nn,:]
                Q2_i = Q2[nn,:]
                ell, pow = self.cal_Cl(Q1_i, Q2_i)
                spectra = np.vstack((ell, pow))
                if ite == 0:
                    spectra_N = spectra[None, :]
                else:
                    spectra_N = np.vstack((spectra_N, spectra[None, :]))
            pbar.set_description('Calculating power spectra')
            pbar.update(1)

        return spectra_N

