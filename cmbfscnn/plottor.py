
# -*- coding: utf-8 -*-

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_sphere_map(denoise_map, target_map, title=[], range=[], save_dir = '',N_sample= 0):
    residual_map = target_map - denoise_map
    fig = plt.figure(1, figsize=(18, 6))
    cmap = plt.get_cmap(plt.cm.jet)
    hp.mollview(target_map[N_sample,:], fig=fig.number, cmap=cmap, sub=(1, 3, 1),
                unit=r'$\mathrm{\mu K}$',min=-1*range[0], max=range[0],title=title[0])
    hp.mollview(denoise_map[N_sample,:], fig=fig.number, cmap=cmap, sub=(1, 3, 2),
                unit=r'$\mathrm{\mu K}$', min=-1*range[1], max=range[1],title=title[1])
    hp.mollview(residual_map[N_sample,:], fig=fig.number, cmap=cmap, sub=(1, 3, 3),
                unit=r'$\mathrm{\mu K}$',min=-1*range[2], max=range[2], title=title[2])
    plt.subplots_adjust(top=0.97, bottom=0.06, left=0.1, right=0.97, hspace=0.01, wspace=0)
    # if not save_dir == None:
    plt.savefig(save_dir+'.png')
    # plt.show()


def plot_image(denoise_map, target_map, title, N_sample= 0,save_dir = '',range=[]):
    denoise_map, target_map = denoise_map[N_sample,:], target_map[N_sample,:]

    fig = plt.figure(figsize=(24, 24))
    gs0 = gridspec.GridSpec(2, 2, figure=fig)
    gs01 = gs0[0].subgridspec(1, 1)
    gs02 = gs0[1].subgridspec(1, 1)
    gs03 = gs0[2].subgridspec(1, 1)
    ax1 = fig.add_subplot(gs01[0])
    ax2 = fig.add_subplot(gs02[0],sharex=ax1,sharey=ax1)
    ax3 = fig.add_subplot(gs03[0],sharex=ax1,sharey=ax1)

    im1 = ax1.imshow(target_map, cmap=plt.cm.jet, vmin=-1*range[0], vmax=range[0])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im1,cax=cax)
    cb.ax.tick_params(which='major',length=12, direction='in', width=3, labelsize=30)
    ax1.set_title(title[0], fontsize=30)


    im2 = ax2.imshow(denoise_map,  cmap=plt.cm.jet, vmin=-1*range[1], vmax=range[1])
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.1)
    cb2 = fig.colorbar(im2,cax=cax2)
    cb2.ax.tick_params(which='major',  length=12, direction='in', width=3, labelsize=30)
    ax2.set_title(title[1], fontsize=30)

    im3 = ax3.imshow(denoise_map-target_map, cmap=plt.cm.jet, vmin=-1*range[2], vmax=range[2])
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im3,cax=cax3)
    cb.ax.tick_params(which='major',  length=12, direction='in', width=3, labelsize=30)
    ax3.set_title(title[2], fontsize=30)

    fig.tight_layout()
    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.05, right=0.94, hspace=0.04, wspace=0.4)
    plt.savefig(save_dir+'.png')
    # plt.show()
    return


def plot_EEBB_PS(ell, out_EE, tar_EE, out_BB, tar_BB, out_denoise_EE, true_EE, out_denoise_BB, true_BB):
    def make_plot(axs):
        box = dict(facecolor='yellow', pad=5, alpha=0.2)

        # Fixing random state for reproducibility

        ax1 = axs[0, 0]
        ax1.plot(ell, tar_EE, label="Simulated noisy EE", c='k')
        ax1.plot(ell, out_EE, label="Recovered noisy EE", c='r')
        ax1.set_ylabel('$D_{\ell}^{EE}$', fontsize=10)
        ax1.set_xlabel('$\ell$', fontsize=10)
        ax1.set_xlim(0, 1500)
        ax1.legend(loc='upper left', fontsize=7)

        ax3 = axs[1, 0]
        ax3.plot(ell, tar_BB, label="Simulated noisy BB", c='k')
        ax3.plot(ell, out_BB, label="Recovered noisy BB", c='r')
        ax3.set_ylabel('$D_{\ell}^{BB}$', fontsize=10)
        ax3.set_xlabel('$\ell$', fontsize=10)
        ax3.set_xlim(0, 1500)
        ax3.legend(loc='upper left', fontsize=7)

        ax2 = axs[0, 1]
        ax2.plot(ell, true_EE, label="True EE", c='k')
        ax2.plot(ell, out_denoise_EE, label="Recovered EE", c='r')
        ax2.set_ylabel('$D_{\ell}^{EE}$', fontsize=10)
        ax2.set_xlabel('$\ell$', fontsize=10)
        ax2.set_xlim(0, 1500)
        ax2.legend(loc='upper left', fontsize=7)

        ax4 = axs[1, 1]
        ax4.plot(ell, true_BB, label="True BB", c='k')
        ax4.plot(ell, out_denoise_BB, label="Recovered BB", c='r')
        ax4.set_ylabel('$D_{\ell}^{BB}$', fontsize=10)
        ax4.set_xlabel('$\ell$', fontsize=10)
        ax4.set_xlim(0, 1500)
        ax4.legend(loc='upper left', fontsize=7)

    # Plot 1:
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6, hspace=0.6)
    make_plot(axs)

    # just align the last column of Axes:
    fig.align_ylabels(axs[:, 1])
    plt.savefig('power.png')
    # plt.show()



def plot_QQUU_PS(ell, out_QQ, tar_QQ, out_UU, tar_UU, out_denoise_QQ, true_QQ, out_denoise_UU, true_UU):
    def make_plot(axs):
        box = dict(facecolor='yellow', pad=5, alpha=0.2)

        # Fixing random state for reproducibility

        ax1 = axs[0, 0]
        ax1.plot(ell, tar_QQ, label="Simulated noisy QQ", c='k')
        ax1.plot(ell, out_QQ, label="Recovered noisy QQ", c='r')
        ax1.set_ylabel('$D_{\ell}^{QQ}$', fontsize=10)
        ax1.set_xlabel('$\ell$', fontsize=10)
        ax1.set_xlim(0, 1500)
        ax1.legend(loc='upper left', fontsize=7)

        ax3 = axs[1, 0]
        ax3.plot(ell, tar_UU, label="Simulated noisy UU", c='k')
        ax3.plot(ell, out_UU, label="Recovered noisy UU", c='r')
        ax3.set_ylabel('$D_{\ell}^{UU}$', fontsize=10)
        ax3.set_xlabel('$\ell$', fontsize=10)
        ax3.set_xlim(0, 1500)
        ax3.legend(loc='upper left', fontsize=7)

        ax2 = axs[0, 1]
        ax2.plot(ell, true_QQ, label="True QQ", c='k')
        ax2.plot(ell, out_denoise_QQ, label="Recovered QQ", c='r')
        ax2.set_ylabel('$D_{\ell}^{QQ}$', fontsize=10)
        ax2.set_xlabel('$\ell$', fontsize=10)
        ax2.set_xlim(0, 1500)
        ax2.legend(loc='upper left', fontsize=7)

        ax4 = axs[1, 1]
        ax4.plot(ell, true_UU, label="True UU", c='k')
        ax4.plot(ell, out_denoise_UU, label="Recovered UU", c='r')
        ax4.set_ylabel('$D_{\ell}^{BB}$', fontsize=10)
        ax4.set_xlabel('$\ell$', fontsize=10)
        ax4.set_xlim(0, 1500)
        ax4.legend(loc='upper left', fontsize=7)

    # Plot 1:
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.2, wspace=0.6, hspace=0.6)
    make_plot(axs)

    # just align the last column of Axes:
    fig.align_ylabels(axs[:, 1])
    plt.savefig('power.png')
    # plt.show()