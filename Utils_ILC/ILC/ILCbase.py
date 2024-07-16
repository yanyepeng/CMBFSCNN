#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .. import smoothingTo
import healpy as hp
import os
import pandas as pd
import pymaster as nmt
eps = 1e-15
def myinv(a):
    if np.shape(a) == 2:
        return np.linalg.inv(a)
    res = np.zeros_like(a)
    for i in range(len(a)):
        try:
            res[i] = np.linalg.inv(a[i])
        except:
            pass
    return res

class baseILC:

    def __init__(self, maps, init_beams, final_beam, *, mask=None):
        '''
        Edited by Si-Yu Li.
        beams are in arcmin
        '''
        self.npix = maps.shape[-1]
        self.nside = hp.npix2nside(self.npix)
        self.nmap = maps.shape[0]
        
        if np.ndim(maps) == 3:
            self.ncomp = maps.shape[1]
            pol = True
        else:
            self.ncomp = 1
            pol = False

        # print('ILC instance initialization')
        # print('nmap:', self.nmap)
        # print('npix:', self.npix)
        # print('ncomp:', self.ncomp)
        
        #init_beams = np.deg2rad(init_beams)/60
        #final_beam = np.deg2rad(final_beam)/60
        #diff_beams = init_beams - final_beam
        #is_smooth = ~(np.round(diff_beams, 6) == 0)

        if mask is None:
            mask = np.ones(self.npix)
        self.mask = mask
        new_map = maps*mask

        for i in range(self.nmap):

            m = maps[i]
            inpbeam = init_beams[i]
            
            inpbeam_isarr = hasattr(inpbeam, '__iter__')
            outbeam_isarr = hasattr(final_beam, '__iter__')
        
            outbeam = np.deg2rad(final_beam/60) if not outbeam_isarr else final_beam
            if not inpbeam_isarr: inpbeam = np.deg2rad(inpbeam/60)

            if inpbeam_isarr and not outbeam_isarr:
                ndim = np.ndim(inpbeam)
                lmax = np.shape(inpbeam)[-1]-1
                if ndim == 1:
                    outbeam = hp.gauss_beam(outbeam, lmax=lmax)
                elif ndim == 2:
                    ncomp = np.shape(inpbeam)
                    if ncomp == 1:
                        inpbeam = inpbeam[0]
                        outbeam = hp.gauss_beam(outbeam, lmax=lmax)
                    elif ncomp == 4:
                        outbeam = hp.gauss_beam(outbeam, lmax=lmax, pol=True)
                    else:
                        raise ValueError('Inp or Out beam has wrong shape.')
                else:
                    raise ValueError('Inp beam has wrong dimension.')
            elif not inpbeam_isarr and outbeam_isarr:
                ndim = np.ndim(outbeam)
                lmax = np.shape(outbeam)[-1]-1
                if ndim == 1:
                    inpbeam = hp.gauss_beam(inpbeam, lmax=lmax)
                elif ndim == 2:
                    ncomp = np.shape(outbeam)
                    if ncomp == 1:
                        outbeam = outbeam[0]
                        inpbeam = hp.gauss_beam(inpbeam, lmax=lmax)
                    elif ncomp == 4:
                        inpbeam = hp.gauss_beam(inpbeam, lmax=lmax, pol=True)
                    else:
                        raise ValueError('Inp or Out beam has wrong shape.')
                else:
                    raise ValueError('Out beam has wrong dimension.')
            
            if inpbeam_isarr or outbeam_isarr:
                assert(np.ndim(inpbeam) == np.ndim(outbeam))
                if np.all(np.isclose(inpbeam, outbeam, rtol=1e-3)):
                    continue
            else:
                if np.isclose(inpbeam, outbeam, rtol=1e-3): continue
            
            new_map[i] = smoothingTo(new_map[i], inpbeam, outbeam)
            
        #for index, flag in enumerate(is_smooth):
        #    if not flag:
        #        continue
        #    new_map[index] = smoothingTo(new_map[index], init_beams[index],
        #                                 final_beam)
        self.maps = new_map
        binmask = (mask != 0)
        self.binmask = binmask
        self.maps[...,~binmask] = 0
        self.final_beam = final_beam

    def map2ilc(self, **kwargs):
        pass

    def ilc2map(self, **kwargs):
        pass

    def calc_weight(self, **kwargs):
        pass

    def do_ilc(self, **kwargs):
        self.final_map = None
        self.map2ilc(self.maps, **kwargs)
        if not hasattr(self, 'weights'):
            self.calc_weight()
        self.final_map = self.ilc2map()
        self.final_map[...,~self.binmask] = 0
        return self.final_map

    def set_weight(self, _ilcbase):
        if hasattr(_ilcbase, 'weights'):
            self.weights = _ilcbase.weights
        else:
            self.weights = _ilcbase


class pixelILC(baseILC):
    def __init__(self, maps, init_beams, final_beam, *, mask=None):
        super().__init__(maps, init_beams, final_beam, mask=mask)

    def map2ilc(self, maps, attr='ilcs', **kwargs):
        mask = self.binmask
        setattr(self, attr, maps[...,mask])

    def ilc2map(self, attr='ilcs'):
        final_map = np.zeros(self.npix)
        final_map[self.binmask] = self.weights @ getattr(self, attr)
        return final_map

    def calc_weight(self):
        R = np.cov(self.ilcs)
        invR = np.linalg.inv(R)
        oneVec = np.ones(self.nmap)
        w = (oneVec@invR)/(oneVec@invR@oneVec)
        self.weights = w

class harmonicILC(baseILC):
    def __init__(self, maps, init_beams, final_beam, *, mask=None):
        # super().__init__(maps, init_beams, final_beam, mask=mask)
        self.npix = maps.shape[-1]
        self.nside = hp.npix2nside(self.npix)
        self.nmap = maps.shape[0]
        if mask is None:
            mask = np.ones(self.npix)
        self.mask = mask
        self.maps = maps*mask
        self.binmask = (mask != 0)
        self.final_beam = np.deg2rad(final_beam)/60

        init_beam_rad = np.deg2rad(init_beams)/60
        self.bl = []
        for i in range(self.nmap):
            self.bl.append(hp.gauss_beam(init_beam_rad[i], 2000, True)[:,1])
    def map2ilc(self, maps, attr='ilcs', mask=None, lmax=400):

        final_bl  = hp.gauss_beam(self.final_beam, 2000, True)[:,1]
        ilcs = []
        for i in range(self.nmap):
            cur_alm = hp.map2alm(maps[i], lmax=lmax)
            cur_alm = hp.almxfl(cur_alm, final_bl/self.bl[i])
            ilcs.append(cur_alm)
        self.lmax = lmax
        setattr(self, attr, ilcs)

    def ilc2map(self):
        final_alm = 0
        for i in range(self.nmap):
            weighted_alm = hp.almxfl(self.ilcs[i], self.weights[i])
            final_alm = final_alm + weighted_alm
            # final_alm = final_alm + hp.almxfl(self.ilcs[i], self.weights[i])
        final_map = hp.alm2map(final_alm, self.nside)
        return final_map

    def calc_weight(self):
        R = np.empty((self.lmax+1, self.nmap, self.nmap))
        for i in range(self.nmap):
            for j in range(self.nmap):
                R[:, i, j] = hp.alm2cl(self.ilcs[i], self.ilcs[j])

        #invR = np.linalg.inv(R[2:])
        invR = myinv(R[2:])
        oneVec = np.ones(self.nmap)
        wl_2 = (oneVec@invR).T/(oneVec@invR@oneVec + eps)
        wl = np.zeros((self.nmap, self.lmax + 1))
        wl[:,2:] = wl_2
        self.weights = wl

data_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)), "needlet_data")
class NILC(baseILC):
    def __init__(self, maps, init_beams, final_beam, *, mask=None):
        super().__init__(maps, init_beams, final_beam, mask=mask)
        self.needlet = pd.read_csv(os.path.join(data_dir, "needlet.csv"))
        self.n_needlet = len(self.needlet)

    def set_needlet(self, path):
        self.needlet = pd.read_csv(path)
        self.n_needlet = len(self.needlet)

    def __calc_hl(self, lmax):
        hl = np.zeros((self.n_needlet, lmax+1))
        for i in range(self.n_needlet):
            nlmax = self.needlet.at[i,'lmax']
            nlmin = self.needlet.at[i,'lmin']
            nlpeak = self.needlet.at[i,'lpeak']
            def funhl(l):
                if l < nlmin or l > nlmax:
                    return 0
                elif l < nlpeak:
                    return np.cos(((nlpeak-l)/(nlpeak-nlmin)) * np.pi/2)
                elif l > nlpeak:
                    return np.cos(((l-nlpeak)/(nlmax-nlpeak)) * np.pi/2)
                else:
                    return 1
            vecHl = np.vectorize(funhl, otypes=[float])
            hl[i] = vecHl(np.arange(lmax+1))
        self.hl = hl

    def map2ilc(self, maps, attr='ilcs', **kwargs):
        lmax = np.max(self.needlet['lmax'])
        #lmax = kwargs.get('lmax', lmax)
        if not hasattr(self, 'hl'):
            self.__calc_hl(lmax)
        hl = self.hl

        alms = []
        for i in range(self.nmap):
            alm = hp.map2alm(maps[i], lmax=lmax)
            alms.append(alm)

        betaList = []
        for j in range(len(self.needlet)):
            curNside = self.needlet.at[j, 'nside']
            curNpix = hp.nside2npix(curNside)
            curBeta = np.zeros((self.nmap, curNpix))
            for i in range(self.nmap):
                curAlm = hp.almxfl(alms[i], hl[j])
                curBeta[i] = hp.alm2map(curAlm, curNside)
            betaList.append(curBeta)

        setattr(self, attr, betaList)

    def __calc_R(self, size = 5):
        betas = self.ilcs
        R = []
        if not hasattr(size, '__len__'):
            size = [size] * self.n_needlet
        for j in range(self.n_needlet):
            curNside = self.needlet.at[j, 'nside']
            curR = np.zeros((hp.nside2npix(curNside), self.nmap, self.nmap))
            for c1 in range(self.nmap):
                for c2 in range(c1+1):
                    #print((c1, c2), end='\t')
                    prodMap = betas[j][c1] * betas[j][c2]
                    RMap = hp.smoothing(prodMap, np.deg2rad(size[j]))
                    curR[:,c1,c2] = RMap
                    curR[:,c2,c1] = RMap
                #print()

            R.append(curR)
        self.R = R
    def calc_weight(self, **kwargs):
        oneVec = np.ones(self.nmap)
        nside = np.array(self.needlet['nside'])
        #size = hp.nside2resol(nside, True) / 60 * 35
        size = 360/(self.needlet['lpeak'] + self.needlet['lmax']) * 10
        #  size = np.arccos(1 - 200/(nside**2))
        #  size = np.rad2deg(size)*0.5
        self.__calc_R(size = size, **kwargs)
        R = self.R
        w = []
        for j in range(self.n_needlet):
            curR = R[j]
            invR = np.linalg.inv(curR)
            curW = (invR@oneVec).T/(oneVec@invR@oneVec)
            w.append(curW)
        self.weights = w

    def ilc2map(self):
        betaNILC = []
        for j in range(self.n_needlet):
            curBeta = self.ilcs[j]
            curW    = self.weights[j]
            curRes  = np.sum(curBeta * curW, axis=0)
            betaNILC.append(curRes)

        resMap = 0
        for j in range(self.n_needlet):
            curAlm = hp.map2alm(betaNILC[j])
            curAlm = hp.almxfl(curAlm, self.hl[j])
            curMap = hp.alm2map(curAlm, self.nside)
            resMap = resMap + curMap
        resMap[~self.binmask] = 0
        return resMap

class LPILC:
    def __init__(self, maps, init_beams, final_beam, *, smoothingmask=None):
        self.npix = maps.shape[-1]
        self.nside = hp.npix2nside(self.npix)
        self.nmap = maps.shape[0]

        if smoothingmask is None:
            smoothingmask = np.ones(self.npix)

        self.init_beams = np.deg2rad(init_beams)/60
        self.final_beam = np.deg2rad(final_beam)/60
        self.smoothed_map = np.zeros_like(maps)
        for i in range(self.nmap):
            self.smoothed_map[i] = smoothingTo(maps[i], self.init_beams[i], self.final_beam)

        self.maps = maps
    def map2ilc(self, maps, mask, attr='ilcs', **kwargs):
        self.ilcmask = mask
        smoothed_mask = nmt.mask_apodization(mask, 3, "C2")
        almes = []
        almbs = []
        for i in range(self.nmap):
            curmap = self.maps[i] * smoothed_mask
            _, alme, almb = hp.map2alm(curmap, lmax=300)
            curbl  = hp.gauss_beam(self.init_beams[i], pol=1)[:,1]
            alme = hp.almxfl(alme, 1/curbl)
            almb = hp.almxfl(almb, 1/curbl)
            almes.append(alme)
            almbs.append(almb)

        almes = np.array(almes)
        almbs = np.array(almbs)
        setattr(self, attr, (almes, almbs))

    def calc_weight(self):
        from scipy.optimize import minimize

        almes, almbs = self.ilcs
        def costFunc(weight):
            wr, wi = np.split(weight, 2)
            outAlmE = wr@almes - wi@almbs
            outAlmB = wr@almbs + wi@almes
            cle = hp.alm2cl(outAlmE)
            clb = hp.alm2cl(outAlmB)
            l = np.arange(cle.size)

            return np.sum(l**2*clb)

        cons = [
            {"type":"eq", "fun":lambda x:np.sum(x[:self.nmap])-1},
            {"type":"eq", "fun":lambda x:np.sum(x[self.nmap:])-0}
        ]
        wGuess = np.ones(2*self.nmap)
        res = minimize(costFunc, wGuess, constraints=cons, callback=lambda x:print(costFunc(x)))

        wr, wi = np.split(res.x, 2)
        self.weights = (wr, wi)

    def ilc2map(self):
        q = self.smoothed_map[:,1,:]
        u = self.smoothed_map[:,2,:]
        wr, wi = self.weights
        xplus = q + 1j*u
        cleanedXplus = (wr+1j*wi)@xplus
        cleanedQ = np.real(cleanedXplus)
        cleanedU = np.imag(cleanedXplus)
        return cleanedQ, cleanedU

    def do_ilc(self, **kwargs):
        self.final_map = None
        self.map2ilc(self.maps, **kwargs)
        if not hasattr(self, 'weights'):
            self.calc_weight()
        self.final_map = self.ilc2map()
        self.final_map[0][~self.ilcmask] = 0
        self.final_map[1][~self.ilcmask] = 0
        return self.final_map

    def set_weight(self, _ilcbase):
        self.weights = _ilcbase.weights

class PILC(baseILC):
    @staticmethod
    def shiftqu(q, u):
        meanq = np.mean(q, 1)
        meanu = np.mean(u, 1)

        shiftedq = q - meanq[...,None]
        shiftedu = u - meanu[...,None]
        return shiftedq, shiftedu

    def __init__(self, maps, init_beams, final_beam, *, mask=None):
        super().__init__(maps, init_beams, final_beam, mask=mask)

    def map2ilc(self, maps, attr='ilcs', **kwargs):
        mask = self.mask
        setattr(self, attr, maps[:,1:])

    def calc_weight(self):
        #q, u = self.shiftqu(self.ilcs[:,0], self.ilcs[:,1])
        q, u = self.ilcs[:,0], self.ilcs[:,1]
        n, npix = self.nmap, self.npix

        cplus  = (q@q.T + u@u.T)/npix
        cminus = (q@u.T - u@q.T)/npix

        C = np.block([[cplus, -cminus], [cminus, cplus]])
        invC = np.linalg.inv(C)

        splus  = np.sum(invC[:n, :n])
        sminus = np.sum(invC[:n, n:])

        lambdaR = 2 * splus/(splus**2 + sminus**2)
        lambdaI = 2 * sminus/(splus**2 + sminus**2)

        wr = np.sum(invC[:n,:n], axis=0) * lambdaR / 2 + np.sum(invC[:n,n:], axis=0) * lambdaI / 2
        wi = np.sum(invC[n:,:n], axis=0) * lambdaR / 2 + np.sum(invC[n:,n:], axis=0) * lambdaI / 2
        self.weights = wr, wi

    def ilc2map(self, attr='ilcs'):
        wr, wi = self.weights
        xplus = self.ilcs[:,0] + 1j*self.ilcs[:,1]
        cleanedXplus = (wr+1j*wi)@xplus
        cleanedQ = np.real(cleanedXplus)
        cleanedU = np.imag(cleanedXplus)
        return np.vstack((cleanedQ, cleanedU))
