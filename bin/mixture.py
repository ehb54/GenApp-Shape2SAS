import argparse
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from shape2sas_helpfunctions import *

if __name__ == "__main__":

    ### input arguments
    parser = argparse.ArgumentParser(description='Compare results from Shape2SAS')
    parser.add_argument('-m', '--model_names',help='Model names')
    parser.add_argument('-f', '--fractions',help='relative fractions of populations',default=False)

    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, help='include flag (no input) to output high resolution plot.')
    parser.add_argument('-s', '--scale', action='store_true', default=False,help='include flag (no input) to scale the simulated intensity of each model in the plots to avoid overlap')    
    parser.add_argument('-n', '--name', help='output filename', default=None)   
    parser.add_argument('-g', '--grid',action='store_true',help='add grid in 2D point representation',default=False)
    parser.add_argument('-norm', '--normalization',help='normalization method: max, I0 (default) or none ',default='max')
    parser.add_argument('-ss', '--sesans', action='store_true',help='plot SESANS data',default=False)
    parser.add_argument('-expo', '--exposure', type=float, default=500, help='Exposure time in arbitrary units.')
    args = parser.parse_args()

    ### colors, models, fractions
    colors = ['blue','red','green','orange','purple','cyan','magenta','grey','pink','forrestgreen']
    models = re.split('[ ,]+', args.model_names)
    fractions = [float(f) for f in re.split('[ ,]+', args.fractions)]
    fractions /= np.sum(fractions)

    # resolution
    if args.high_res:
        format = '.pdf'
    else:
        format ='.png'

    ### plot SAS data: p(r), I(q), Isim(q)
    fig, ax = plt.subplots(1,3,figsize=(12,4))
    scale_factor = 1
    zo=1
    all_model_names = ''
    w_list,q_list,I_list,r_list,pr_list,I0_list = [],[],[],[],[],[]
    dmax = 0
    for i,model in enumerate(models):
        pr_filename = model + '/pr_' + model + '.dat'
        r,pr = np.genfromtxt(pr_filename,skip_header=1,unpack=True)
        if args.normalization in ['I0','Forward_Scattering','I0','I(0)','integral']:
            dr = r[4]-r[3]
            pr /= pr.sum()*dr
        elif args.normalization in ['max','Max','pr_max','prmax']:
            pr /= np.max(pr)
        elif args.normalization in ['none','no','None','No']:
            pass
        else: 
            print('\n\nERROR: unknown normalization argument: ' + args.normalization + '. Should be "max" or "I0" or "none".\n\n')
            exit()
        ax[0].plot(r,pr,color=colors[i],label=model)

        Iq_filename = model + '/Iq_' + model + '.dat'
        q,I = np.genfromtxt(Iq_filename,skip_header=2,unpack=True)
        ax[1].plot(q,I,color=colors[i],label=model)

        Isim_filename = model + '/Isim_' + model + '.dat'
        q,Isim,sigma = np.genfromtxt(Isim_filename,skip_header=3,unpack=True)
        if args.scale: 
            ax[2].errorbar(q,Isim*scale_factor,yerr=sigma*scale_factor,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s, scaled by %1.0e' % (model,scale_factor),zorder=1/zo)
            scale_factor *= 0.1
        else:
            ax[2].errorbar(q,Isim,yerr=sigma,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s' % model,zorder=zo)
        if i > 0:
            all_model_names += '_'
        all_model_names += model
        w_list.append(fractions[i] * np.sqrt(Isim[0]))
        q_list.append(q)
        I_list.append(I)
        r_list.append(r)
        pr_list.append(pr)
        I0_list.append(Isim[0])

        if np.max(r) > dmax:
            dmax = np.max(r)

    # prepare output folder
    if args.name == None:
        name = all_model_names
    else:
        name = args.name
    folder = 'mixture_' + name

    r_mix = np.linspace(0,dmax,100)
    pr_mix = np.zeros_like(r)
    w_sum,I_mix,pr_mix,I0_mix = 0,0,0,0
    for i,w in enumerate(w_list):
        w_sum += w
        I_mix +=  w * I_list[i]
        pr_mix += w * np.interp(r_mix,r_list[i],pr_list[i],left=0,right=0)
        I0_mix += w * I0_list[i]
        # obs: I may not be same lengths!!
    if args.normalization in ['I0','Forward_Scattering','I0','I(0)','integral']:
        dr = r_mix[4]-r_mix[3]
        pr_mix /= pr_mix.sum()*dr
    elif args.normalization in ['max','Max','pr_max','prmax']:
        pr_mix /= np.max(pr_mix)
    elif args.normalization in ['none','no','None','No']:
        pass
    else: 
        print('\n\nERROR: unknown normalization argument: ' + args.normalization + '. Should be "max" or "I0" or "none".\n\n')
        exit()
    save_pr_func(r,pr_mix,folder)
    
    I_mix /= w_sum
    I0_mix /= w_sum
    save_I_func(q,I_mix,folder)

    ax[0].plot(r,pr_mix,color='black',label='mixture')
    ax[1].plot(q,I_mix,color='black',label='mixture')

    #### Simulate I(q) 
    Isim_mix,sigma_mix = simulate_data_func(q,I_mix,I0_mix,args.exposure)
    save_Isim_func(q,Isim_mix,sigma_mix,folder)

    if args.scale: 
        ax[2].errorbar(q,Isim_mix*scale_factor,yerr=sigma_mix*scale_factor,linestyle='none',marker='.', color='black',label=r'$I_\mathrm{sim}(q)$, %s, scaled by %1.0e' % ('mixture',scale_factor),zorder=1/zo)
        scale_factor *= 0.1
    else:
        ax[2].errorbar(q,Isim_mix,yerr=sigma_mix,linestyle='none',marker='.', color='black',label=r'$I_\mathrm{sim}(q)$, %s' % 'mixture',zorder=zo)

    ax[0].set_xlabel(r'$r$ [$\mathrm{\AA}$]')
    ax[0].set_ylabel(r'$p(r)$')
    ax[0].set_title('pair distance distribution function')
    ax[0].legend(frameon=False)

    if not args.xscale_lin:
        ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[1].set_ylabel(r'normalized $I(q)$')
    ax[1].set_title('normalized scattering, no noise')
    ax[1].legend(frameon=False)

    if not args.xscale_lin:
        ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$q$ [$\mathrm{\AA}^{-1}$]')
    ax[2].set_ylabel(r'$I(q)$ [a.u.]')
    ax[2].set_title('simulated scattering, with noise')
    ax[2].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(folder + '/' + folder + format)

    ### plot sesans data, G(delta), G_sim(delta) - if opted for
    if args.sesans:

        fig, ax = plt.subplots(1,2,figsize=(8,4))
        scale_factor = 1
        for i,model in enumerate(models):
            G_filename = model + '/G_' + model + '.ses'
            d,G = np.genfromtxt(G_filename,skip_header=2,unpack=True)
            ax[0].plot(d,G,color=colors[i],label=model)

            ax[0].set_ylabel(r'$G(\delta)$ [cm$^{-1}$]')
            ax[0].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
            ax[0].set_title('theoretical SESANS, no noise')
            ax[0].legend(frameon=False)
        
            Gsim_filename = model + '/Gsim_' + model + '.ses'
            d,Gsim,sigmaG = np.genfromtxt(Gsim_filename,skip_header=2,unpack=True)
            if args.scale: 
                ax[1].errorbar(d,Gsim*scale_factor,yerr=sigmaG*scale_factor,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s, scaled by %1.0e' % (model,scale_factor),zorder=1/zo)
                scale_factor *= 0.1
            else:
                ax[1].errorbar(d,Gsim,yerr=sigmaG,linestyle='none',marker='.', color=colors[i],label=r'$I_\mathrm{sim}(q)$, %s' % model,zorder=zo)
            if i > 0:
                all_model_names += '_'
            all_model_names += model

            ax[1].set_xlabel(r'$\delta$ [$\mathrm{\AA}$]')
            ax[1].set_ylabel(r'$\ln(P)/(t\lambda^2)$ [$\mathrm{\AA}^{-2}$cm$^{-1}$]')
            ax[1].set_title('simulated SESANS, with noise')
            ax[1].legend(frameon=True)

        plt.tight_layout()
        plt.savefig(folder + '/' + folder + '_sesans' + format)

    plt.show()




