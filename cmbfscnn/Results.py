
# -*- coding: utf-8 -*-
import numpy as np
from . import plottor as pt

from . import CMBFS_mode as Cm



class Plot_results(Cm.Calculate_power_spectra):
    def __init__(self,  result_dir = 'DATA_results/', is_half_split_map=True):
        super(Plot_results, self).__init__()
        self.result_dir = result_dir
        self.is_half_split_map = is_half_split_map
        self.save_PS_dir
        self._creat_ps_file

    def plot_predicted_sphere_map(self):
        if self.is_half_split_map:
            pre_cmbQ = np.load(getattr(self,'output_Q_dir_1'))
            target_cmbQ = np.load(getattr(self,'target_Q_dir_1'))
            pre_cmbU = np.load(getattr(self, 'output_U_dir_1'))
            target_cmbU = np.load(getattr(self, 'target_U_dir_1'))
        else:
            pre_cmbQ = np.load(getattr(self, 'output_Q_dir'))
            target_cmbQ = np.load(getattr(self, 'target_Q_dir'))
            pre_cmbU = np.load(getattr(self, 'output_U_dir'))
            target_cmbU = np.load(getattr(self, 'target_U_dir'))
        pt.plot_sphere_map(pre_cmbQ, target_cmbQ, title=['Simulated CMB Q map', 'Recovered CMB Q map', 'Residual'],
                        range=[10, 10, 0.2],save_dir='recover_CMB_Q_map',N_sample=0)
        pt.plot_sphere_map(pre_cmbU, target_cmbU, title=['Simulated CMB U map', 'Recovered CMB U map', 'Residual'],
                        range=[10, 10, 0.2], save_dir='recover_CMB_U_map',N_sample=0)

    def plot_predicted_flat_map(self):
        if self.is_half_split_map:
            pre_cmbQ = np.load(getattr(self, 'output_Qmap_dir') + 'predicted_CMB_Q'  + '_map_half_1.npy')
            target_cmbQ = np.load(getattr(self, 'output_Qmap_dir') + 'target_CMB_Q'  + '_map_half_1.npy')
            pre_cmbU = np.load(getattr(self, 'output_Umap_dir') + 'predicted_CMB_U' + '_map_half_1.npy')
            target_cmbU = np.load(getattr(self, 'output_Umap_dir') + 'target_CMB_U' + '_map_half_1.npy')
        else:
            pre_cmbQ = np.load(getattr(self, 'output_Qmap_dir') + 'predicted_CMB_Q' + '_map.npy')
            target_cmbQ = np.load(getattr(self, 'output_Qmap_dir') + 'target_CMB_Q' + '_map.npy')
            pre_cmbU = np.load(getattr(self, 'output_Umap_dir') + 'predicted_CMB_U' + '_map.npy')
            target_cmbU = np.load(getattr(self, 'output_Umap_dir') + 'target_CMB_U' + '_map.npy')
        title1,title2 = ['Simulated CMB Q map', 'Recovered CMB Q map', 'Residual'], ['Simulated CMB U map', 'Recovered CMB U map', 'Residual']
        pt.plot_image(pre_cmbQ, target_cmbQ, title1, N_sample=0, save_dir='recovered_CMB_flat_Qmap', range=[10,10,0.2])
        pt.plot_image(pre_cmbU, target_cmbU, title1, N_sample=0, save_dir='recovered_CMB_flat_Umap', range=[10, 10, 0.2])

    def plot_recovered_CMB_QU_PS(self,nlb=5):
        pre_cmbQ_ps = np.load(getattr(self, 'save_output_Q_dir_1').format(nlb))
        pre_cmbU_ps = np.load(getattr(self, 'save_output_U_dir_1').format(nlb))
        tar_cmbQ_ps = np.load(getattr(self, 'save_target_Q_dir_1').format(nlb))
        tar_cmbU_ps = np.load(getattr(self, 'save_target_U_dir_1').format(nlb))
        true_cmb_Q_ps = np.load(getattr(self, 'true_output_Q_dir').format(nlb))
        true_cmb_U_ps = np.load(getattr(self, 'true_output_U_dir').format(nlb))
        pre_denoise_Q_ps = np.load(getattr(self, 'save_output_cros_Q_dir').format(nlb))
        pre_denoise_U_ps = np.load(getattr(self, 'save_output_cros_U_dir').format(nlb))
        print('+++++++++++', pre_cmbQ_ps.shape)
        pt.plot_QQUU_PS(ell=pre_cmbQ_ps[0,0,:], out_QQ=pre_cmbQ_ps[0,1,:], tar_QQ=tar_cmbQ_ps[0,1,:], out_UU=pre_cmbU_ps[0,1,:],
                     tar_UU=tar_cmbU_ps[0,1,:], out_denoise_QQ=pre_denoise_Q_ps[0,1,:], true_QQ=true_cmb_Q_ps[0,1,:],
                     out_denoise_UU=pre_denoise_U_ps[0,1,:], true_UU=true_cmb_U_ps[0,1,:])

    def plot_recovered_CMB_EB_PS(self,nlb=5):
        pre_cmbEB_ps = np.load(getattr(self, 'save_output_EB_dir').format(nlb))
        tar_cmbEB_ps = np.load(getattr(self, 'save_target_EB_dir').format(nlb))
        true_cmbEB_ps = np.load(getattr(self, 'save_true_EB_dir').format(nlb))
        pre_denoise_cmbEB_ps = np.load(getattr(self, 'save_output_cros_EB_dir').format(nlb))
        pt.plot_EEBB_PS(ell=pre_cmbEB_ps[0,0,:], out_EE=pre_cmbEB_ps[0,1,:], tar_EE=tar_cmbEB_ps[0,1,:], out_BB=pre_cmbEB_ps[0,2,:], tar_BB=tar_cmbEB_ps[0,2,:],
                     out_denoise_EE=pre_denoise_cmbEB_ps[0,1,:], true_EE=true_cmbEB_ps[0,1,:], out_denoise_BB=pre_denoise_cmbEB_ps[0,2,:], true_BB=true_cmbEB_ps[0,2,:])










