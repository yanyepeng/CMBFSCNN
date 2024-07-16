import camb
import numpy as np

# this parameters are get from Table 3 (TT+lowP) of Planck 2015 results -XIII
hubble = [67.31, 0.96]   # [best fitting values, std 1 sigma]
ombh2 = [0.02222, 0.00023]
omch2 = [0.1197, 0.0022]
re_optical_depth = [0.078, 0.019]
scalar_amp_1 = [2.1955e-9, 0.0791e-9]
scalar_spectral_index_1 = [0.9655, 0.0062]

def sim_power_spectra(sim_H0, sim_ombh2, sim_omch2, sim_tau, sim_As, sim_ns,
                      spectra_type='lensed_scalar'):
    '''
    spectra_type: 'lensed_scalar', 'unlensed_total'.
    given five Cosmological parameter ,use camb software package ot get power_sperctra.
    our target is to obtain 'cmb_specs' that is necessary in simulating cmb with pysm.
    obtaining power spectra is fist step for getting 'cmb_specs'
    '''
    #Set up a new set of parameters for CAMB
    #pars = camb.CAMBparams()
    pars = camb.model.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and \
    #helium set using BBN consistency
    pars.set_cosmology(H0=sim_H0, ombh2=sim_ombh2, omch2=sim_omch2, tau=sim_tau)
    pars.InitPower.set_params(As=sim_As, ns=sim_ns)
    pars.set_for_lmax(3500, lens_potential_accuracy=1);
    #calculate results for these parameters
    results = camb.get_results(pars)
    cl_phi = results.get_lens_potential_cls(lmax=3550)
    #print('phi power', cl_phi.shape)
    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars)
    # unlensedtotCls = powers['unlensed_total']
    unlensedtotCls = powers[spectra_type]
    #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries
    #will be zero by default.
    #The different CL are always in the order TT, EE, BB, TE (with BB=0 for
    #unlensed scalar results).
    CMB_outputscale = 7.42835025e12
    unlensedtotCls = unlensedtotCls*CMB_outputscale
    l = np.arange(unlensedtotCls.shape[0])[2:]
    unlensedtotCls = unlensedtotCls[2:, :]
    index = unlensedtotCls.shape[0]
    cl_phi = cl_phi[2:,:]  # phi phi spectrum
    cl_phi = cl_phi[:index,:]
    #print('phi power', cl_phi)
    unlensedtotCls_ = np.c_[l, unlensedtotCls,cl_phi]
    #print('=========', unlensedtotCls_[:].shape)
    return unlensedtotCls_

def sim_UniformParam(param, times_l = 5, times_r = 5):
    range_param = [param[0]-param[1]*times_l, param[0]+param[1]*times_r]
    sim_param = np.random.rand() * (range_param[1] - range_param[0]) + range_param[0]
    return sim_param

def sim_NormalParams(param, param_label):
    sim_param = np.random.randn() * param[1] + param[0]
#    print '%s:'%param_label, '%s'%sim
    return sim_param

# this function is randomizing cosmological parameters
# then using random or fixed  cosmological parameters to  get power pectra
# if random is 'Normal', paramater 'times' is useless
def get_spectra(random='Normal', times=5, spectra_type='lensed_scalar'):
    if random == 'Normal':
        sim_H0 = sim_NormalParams(hubble, 'hubble')  # realize random for cosmological parameter
        sim_ombh2 = sim_NormalParams(ombh2, 'ombh2')
        sim_omch2 = sim_NormalParams(omch2, 'omch2')
#        sim_tau = sim_NormalParams(re_optical_depth, 're_optical_depth')
        sim_tau = 0.078
        sim_As = sim_NormalParams(scalar_amp_1, 'scalar_amp_1')
        sim_ns = sim_NormalParams(scalar_spectral_index_1, 'scalar_spectral_index_1')
    elif random == 'Uniform':
        sim_H0 = sim_UniformParam(hubble, times_l=times, times_r=times)
        sim_ombh2 = sim_UniformParam(ombh2, times_l=times, times_r=times)
        sim_omch2 = sim_UniformParam(omch2, times_l=times, times_r=times)
#        times_tau_l = (0.078-0.003)/0.019
#        sim_tau = sim_UniformParam(re_optical_depth, times_l=times_tau_l, times_r=times)
        sim_tau = 0.078
        sim_As = sim_UniformParam(scalar_amp_1, times_l=times, times_r=times)
        sim_ns = sim_UniformParam(scalar_spectral_index_1, times_l=times, times_r=times)
    elif random == 'fixed':
        sim_H0 = hubble[0]
        sim_ombh2 = ombh2[0]
        sim_omch2 = omch2[0]
        sim_tau = re_optical_depth[0]
        sim_As = scalar_amp_1[0]
        sim_ns = scalar_spectral_index_1[0]
    sim_params = np.c_[sim_H0,sim_ombh2,sim_omch2,sim_tau,sim_As,sim_ns]
    unlensedtotCls = sim_power_spectra(sim_H0,sim_ombh2,sim_omch2,sim_tau,sim_As,sim_ns, spectra_type=spectra_type)
    return unlensedtotCls, sim_params   # the shape of unlensedtotCls is [N_l, 5]
# [:,0] is ell, index from 1 to 5 represent TT, EE, BB, TE

def ReadClFromPycamb(random=None, times=None, runCAMB = False, spectra_type='lensed_scalar'):
    if runCAMB:
        cls, sim_params = get_spectra(random=random, times=times, spectra_type=spectra_type)
        
        # cls is power spectra wiht shape[N_l,5]. [:,0] is ell, index from 1 to 5 represent TT, EE, BB, TE
        data = Spectra(Cls = cls.transpose(), isCl = False, \
        Name = 'CMB Spectra', Checkl = False) #
        # .transpose() can change shape Cls, make the shape (N_l, 5) of Cls become (5,N_l)
        # Spectra function is checking ell, which didn't seem to help much
    # data.Cls = np.concatenate([data.Cls, np.zeros([3,len(data.Cls[0])])]) ## shape (8,N_l)
    # data.Cls = np.vstack((data.Cls,cl_phi))
    return data, sim_params  #

# this function is checking ell, which didn't seem to help much
class Spectra:
    def __init__(self, Cls = None, isCl = True, Name = '', Checkl = True):
        self.Name = Name
        self.Cls = Cls # cls is power spectra wiht shape[N_l,5]. [:,0] is ell, index from 1 to 5 represent TT, EE, BB, TE
        self.isCl = isCl
        self.isDl = not self.isCl
        if Checkl:
            self.lcheck()
    def lcheck(self, lmax = None):
        if type(self.Cls) != type(None):
            lmin = int(min(self.Cls[0])) # calculate minimum of ell
            if lmin > 0:
                tmp = np.zeros((np.shape(self.Cls)[0] - 1 , lmin))
                tmp1 = np.array([range(lmin)])
                tmp = np.concatenate((tmp1, tmp))
                self.Cls = np.concatenate((tmp, self.Cls), axis = 1)
            if lmax == None:
                lmax = max(self.Cls[0]) # calculate maximum of ell
            lmax = int(lmax)
            self.Cls = self.Cls.transpose()[:lmax + 1].transpose()
            # .transpose() can change shape Cls, make the shape (N_l, 5) of Cls become (5,N_l)
    def Cl2Dl(self):
        if self.isDl:
            raise ValueError('Spectra have already been multiplied by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            data[i][1:] = data[i][1:] * data[i][0] * (data[i][0] + 1.) / 2. / np.pi
        self.Cls = data.transpose()
        self.isDl = True
        self.isCl = False
    def Dl2Cl(self):
        if self.isCl:
            raise ValueError('Spectra have already been divided by factor l(l+1)/2/pi')
        data = self.Cls.transpose()
        for i in range(len(data)):
            if data[i][0] == 0.:
                continue
            data[i][1:] = data[i][1:] / data[i][0] / (data[i][0] + 1.) * 2. * np.pi
        self.Cls = data.transpose()
        self.isDl = False
        self.isCl = True

# this function return the cmb_specs in synfast model of PYSM
def ParametersSampling(random='Normal', times=5, spectra_type='lensed_scalar'):
#        Components.ParametersSampling(self)#original
#        data = ReadClFromCAMB('CMB_ML', params = self.paramsample, runCAMB = True)#original
    data, sim_params = ReadClFromPycamb(random=random, times=times, runCAMB = True, spectra_type=spectra_type) # added
    return data.Cls  # power specs, shape(8,N_l),
# [:,0] is ell, index from 1 to 5 represent TT, EE, BB, TE, index from 5 to 8 are
