#!/usr/bin/env python
# -*- coding: utf-8 -*-

import healpy as hp
import numpy as np

def smoothingTo(inputMap, oriBeam, destBeam):
    '''
    Edited by Si-Yu Li. 2022-05-18
    '''
    npix = np.shape(inputMap)[-1]
    nside = hp.nside2npix(npix)
    lmax = 3*nside+1

    assert(np.ndim(inputMap)<=2)

    haspol = np.ndim(inputMap) == 2

    if not hasattr(oriBeam, '__iter__'):
        oriBeam = float(oriBeam)
        oriBl = hp.gauss_beam(oriBeam, lmax=lmax, pol=haspol)
    else:
        oriBl = oriBeam
    
    if not hasattr(destBeam, '__iter__'):
        destBeam = float(destBeam)
        destBl = hp.gauss_beam(destBeam, lmax=lmax, pol=haspol)
    else:
        destBl = destBeam
    
    ori_lmax_bl = np.shape(oriBl)[-1]
    dest_lmax_bl = np.shape(destBl)[-1]

    lmax = min(ori_lmax_bl, dest_lmax_bl)

    Bl = destBl[:lmax+1] / oriBl[:lmax+1]

    res = hp.smoothing(inputMap, beam_window=Bl, lmax=lmax, pol=haspol)

    return res

#def smoothingTo(inputMap, oriBeam, destBeam):
#    '''in radians'''
#    newfwhm = np.sqrt(destBeam**2 - oriBeam**2)
#    res = hp.smoothing(inputMap, newfwhm)
#    return res

if __name__ == '__main__':
    nside = 128
    npix = hp.nside2npix(nside)
    m = np.random.normal(0,100, npix)
    m1 = hp.smoothing(m, np.deg2rad(0.5))
    m2 = smoothingTo(m1, np.deg2rad(0.5), np.deg2rad(1))

    m3 = hp.smoothing(m, np.deg2rad(1))
    r = hp.anafast(m2)
    r2 = hp.anafast(m3)
    import matplotlib.pyplot as plt
    plt.plot(r)
    plt.plot(r2)
    plt.show()
