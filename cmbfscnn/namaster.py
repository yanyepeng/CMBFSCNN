
# -*- coding: utf-8 -*-

import pymaster as nmt
import numpy as np



class Calculate_power_spectrum(object):
    def __init__(self, map1, map2, mask, nside=512, aposize=None, nlb=10, Bl=None,Dl=True):
        '''
        :param map1:  field 1, spin-1 (1-D array with shape (nside**2*12)) or spin-2 (2-D array). spin-0 field represents T map, and spin-2 field is Q+iU or Q-iU
        :param map2: field 1, spin-1 or spin-2
        :param mask: 1-D array with shape (nside**2*12,), the mask file used to the CMB map.
        :param aposize: float or None, apodization scale in degrees.
        :param nlb: int, the bin size (\delta_\ell) of multipoles, it can be set to ~ 1/fsky
        :param Dl: if True return Dl, if False return Cl
        '''
        self.map1 = map1
        self.map2 = map2
        self.mask = mask
        self.nside = nside
        self.aposize = aposize
        self.nlb = nlb
        self.Bl = Bl
        self.Dl = Dl
    
        
    def cl2dl(self, Cl, ell_start=2, ell_in=None, get_ell=True):
        '''
        calculate Dl from Cl
        ell_start: 0 or 2, which should depend on Dl
        ell_in: the ell of Cl (as the input of this function)
        '''
        if ell_start==0:
            lmax_cl = len(Cl) - 1
        elif ell_start==2:
            lmax_cl = len(Cl) + 1
        
        ell = np.arange(lmax_cl + 1)
        if ell_in is not None:
            if ell_start==2:
                ell[2:] = ell_in
        
        factor = ell * (ell + 1.) / 2. / np.pi
        if ell_start==0:
            Dl = np.zeros_like(Cl)
            Dl[2:] = Cl[2:] * factor[2:]
            ell_2 = ell
        elif ell_start==2:
            Dl = Cl * factor[2:]
            ell_2 = ell[2:]
        if get_ell:
            return ell_2, Dl
        else:
            return Dl
    

    def compute_master(self, f_a, f_b, wsp, clb=None):
        # Compute the power spectrum (a la anafast) of the masked fields
        # Note that we only use n_iter=0 here to speed up the computation,
        # but the default value of 3 is recommended in general.
        cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
        # Decouple power spectrum into bandpowers inverting the coupling matrix
        cl_decoupled = wsp.decouple_cell(cl_coupled, cl_bias=clb)
        return cl_decoupled
    
    
    def get_power_spectra_for_spin1(self):
        '''
        Calculate Cl * ell*(ell+1)/2/np.pi for spin-1 including TT, QQ, and UU.
        '''
        if self.aposize is not None:    # apodize mask on a scale of self.aposize  deg 
            mask = nmt.mask_apodization(self.mask, aposize=self.aposize, apotype="Smooth")
        else:
            mask = self.mask
        # Create contaminated fields
        f_1 = nmt.NmtField(mask, [self.map1], templates=None, beam=self.Bl)
        f_2 = nmt.NmtField(mask, [self.map2], templates=None, beam=self.Bl)
        # Create binning scheme. We will use 20 multipoles per bandpower. nlb=\delta_ell ~ 1/fsky 
        b = nmt.NmtBin.from_nside_linear(self.nside, nlb=self.nlb, is_Dell=False) 
        
        # We then generate an NmtWorkspace object that we use to compute and store
        # the mode coupling matrix. Note that this matrix depends only on the masks
        # of the two fields to correlate, but not on the maps themselves (in this
        # case both maps are the same.
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f_1, f_2, b)
        cl_master = self.compute_master(f_1, f_2, w) # get power spectra Cl 
        ell = b.get_effective_ells()  # get ell
        if self.Dl:
            ell, dl_master = self.cl2dl(cl_master[0], ell_start=2, ell_in=ell) # get Dl
        else:
            ell = ell
            dl_master = cl_master[0]
        return ell, dl_master
    
    def get_power_spectra_for_spin2(self):
        '''
        Calculate Cl * ell*(ell+1)/2/np.pi for spin-2 including EE and BB.
        '''
        if self.aposize is not None:    # apodize mask on a scale of self.aposize  deg 
            mask = nmt.mask_apodization(self.mask, aposize=self.aposize, apotype="Smooth")
        else:
            mask = self.mask
        # Create contaminated fields
        f_1 = nmt.NmtField(mask, self.map1, templates=None, beam=self.Bl)
        f_2 = nmt.NmtField(mask, self.map2, templates=None, beam=self.Bl)
        if self.Dl:
            b = nmt.NmtBin.from_nside_linear(self.nside, nlb=self.nlb, is_Dell=True) 
        else:
            b = nmt.NmtBin.from_nside_linear(self.nside, nlb=self.nlb, is_Dell=False)
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f_1, f_2, b)
        Dl_master = self.compute_master(f_1, f_2, w)
        ell = b.get_effective_ells()
        return ell, Dl_master   # the shape of fl_22 is (4, cl): (EE,cl), (EB, cl), (BE,cl), (BB)
        
































