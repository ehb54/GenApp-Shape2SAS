#!/usr/bin/python3

import time
import argparse
from sys import argv
import numpy as np
import shutil
from shape2sas_helpfunctions import *

# current version
version = 2.5

if __name__ == "__main__":
       
    ### start timing
    start_total = time.time()

    ### remove any existing log file
    open('shape2sas.log','w').close()

    ### welcome message
    printt('#######################################################################################')
    printt('RUNNING shape2sas.py version %s \n - for instructions type: python shape2sas.py -h' % version)
    command = "python shape2sas.py"
    for aa in argv[1:]:
        if ' ' in aa:
            command += " \"%s\"" % aa
        else:
            command += " %s" % aa
    printt('command used: %s' % command)
    printt('#######################################################################################')

    ### input values 
    parser = argparse.ArgumentParser(description='Shape2SaS - calculates small-angle scattering from a given shape defined by the user.')
      
    # mandatory inputs
    parser.add_argument('-s', '--subunit', type=separate_string, nargs='+', action='extend',
                       help='Type of subunits for each model.')
    parser.add_argument('-d', '--dimension', type=float_list, nargs='+', action='append',
                        help='dimensions of subunits for each model.')
    
    # optional model-dependent inputs:
    parser.add_argument('-m', '--model_name', nargs='+', action='extend',
                        help='Name of model.')
    parser.add_argument('-sld', '--sld', type=float, nargs='+', action='append',
                        help='excess scattering length density or contrast.')
    parser.add_argument('-pd', '--polydispersity', type=float, nargs='+', action='extend',
                        help='Polydispersity of subunits for each model.')
    parser.add_argument('-com', '--com', type=float_list, nargs='+', action='append',
                         help='displacement for each subunits in each model.')
    parser.add_argument('-rot', '--rotation', type=float_list, nargs='+', action='append',
                         help='rotation for each subunits in each model.')
    parser.add_argument('-sigmar', '--sigma_r', type=float, nargs='+', action='extend',
                        help='interface roughness for each model.')
    parser.add_argument('-c', '--conc', type=float, nargs='+', action='extend',
                        help='volume fraction concentration.')
    parser.add_argument('-exclude', '--exclude_overlap', type=str2bool, nargs='+', action='extend',
                        help='bool to exclude overlap.')

    # optional structure factor related inputs
    parser.add_argument('-S', '--S', type=str, nargs='+', action='extend',
                        help='structure factor: None/HS/aggregation in each model.')
    parser.add_argument('-Sp', '--S_par', type=float_list, nargs='+', action='append',
                        help='parameters of structure factor for each model.')

    # optional general inputs
    parser.add_argument('-qmin', '--qmin', type=float, default=0.001, 
                        help='Minimum q-value for the scattering curve.')
    parser.add_argument('-qmax', '--qmax', type=float, default=0.5, 
                        help='Maximum q-value for the scattering curve.')
    parser.add_argument('-Nq', '--qpoints', type=int, default=400, 
                        help='Number of points in q.')
    parser.add_argument('-Np', '--prpoints', type=int, default=100, 
                        help='Number of points in the pair distance distribution function.')
    parser.add_argument('-N', '--Npoints', type=int, default=8000, 
                        help='Number of simulated points per model.')
    parser.add_argument('-expo', '--exposure', type=float, default=500, 
                        help='Exposure time in arbitrary units.')

    # optional plot options
    parser.add_argument('-lin', '--xscale_lin', action='store_true', default=False, 
                        help='include flag (no input) to make q scale linear instead of logarithmic.')
    parser.add_argument('-hres', '--high_res', action='store_true', default=False, 
                        help='include flag (no input) to output high resolution plot.')

    # optional SESANS-related options (Shape2SESANS)
    parser.add_argument('-ss', '--sesans', action='store_true', default=False,
                        help='Calculate SESANS data from the SAS data.')
    parser.add_argument('-sse', '--sesans_error', type=float, default=0.02, 
                        help='Baseline SESANS error relative to max signal.')
    parser.add_argument('-Nd', '--deltapoints', type=int, default=150,
                        help='Number of points in delta.')
    
    args = parser.parse_args()

    ### check input 
  
    # check that subunits and dimensions are provided
    if args.subunit is None:
        raise argparse.ArgumentError(args.subunit, "No subunit type was given as an input.")  
    if args.dimension is None:
        raise argparse.ArgumentError(args.dimension, "No dimensions were given as an input.")
    # check that number of subunits matches number of dimension lists
    for subunit, dimension in zip(args.subunit, args.dimension):
         if len(subunit) != len(dimension):
            raise argparse.ArgumentTypeError("Mismatch between number subunit types (%d) and dimensions lists (%d)." % (len(subunit),len(dimension)))    
    num_models = len(args.subunit)
    if num_models == 1:
        printt(f"Simulating {num_models} model...")
    else: 
        printt(f"Simulating {num_models} models...")

    # prepare lists (several models can be simulated simultaneously)
    r_list, pr_norm_list, I_list, I_sim_list, sigma_list, S_list = [], [], [], [], [], [] 
    x_list, y_list, z_list, sld_list, model_filename_list, model_name_list = [], [], [], [], [], []
    if args.sesans:
        delta_list,G_list,G_sim_list,sigma_G_list = [],[],[],[]

    # loop over models
    for i in range(num_models):
        
        ### read and print model name for model i
        model_name = check_input(args.model_name, f"Model {i}", "model name", i)
        if model_name in model_name_list:
            #model names should be unique - else add a number
            model_name += '_' + str(i+1)  
        model_filename = "_".join(model_name.split()) # remove whitespace for filenames
        model_name_list.append(model_name)
        model_filename_list.append(model_filename)

        #### read number of subunits, SLD, COM, rotation and exclude overlap for model i
        N_subunits = len(args.subunit[i])
        sld = check_3Dinput(args.sld, [1.0], "SLD", N_subunits, i)
        com = check_3Dinput(args.com, [[0, 0, 0]], "COM", N_subunits, i)
        rotation = check_3Dinput(args.rotation, [[0, 0, 0]], "rotation", N_subunits, i)
        exclude_overlap = check_input(args.exclude_overlap, True, "exclude_overlap", i)

        ### make point cloud
        printt(f"    Generating points for Model: " + model_name)        
        point_distribution = getPointDistribution(args.subunit[i],sld,args.dimension[i],com,rotation,exclude_overlap,args.Npoints)
        save_points(point_distribution, model_filename)
        x_list.append(np.concatenate(point_distribution.x))
        y_list.append(np.concatenate(point_distribution.y))
        z_list.append(np.concatenate(point_distribution.z))
        sld_list.append(np.concatenate(point_distribution.sld))

        ### read concentration, interface roughness/fuzziness, structure factor and structure factor-related parameters for model i
        conc = check_input(args.conc, 0.02, "concentration", i)
        sigma_r = check_input(args.sigma_r, 0.0, "sigma_r", i)
        S_type = check_input(args.S, 'None', "Structure type", i)
        stype = S_type.lower().replace("_", "").replace(" ", "")
        try:
            S_par = args.S_par[i][0]
        except:
            S_par = []

        ### calculate p(r)
        printt("\n    Calculating pair distance distribution, p(r)...")
        polydispersity = check_input(args.polydispersity, 0.0, "polydispersity", i)
        r, pr, pr_norm, dmax = calc_pr_func(point_distribution,prpoints=args.prpoints, polydispersity=polydispersity)
        save_pr_func(r,pr_norm,model_filename)
        r_list.append(r)
        pr_norm_list.append(pr_norm)

        ### define q (and if sesans is opted for, also define the spin echo length, delta)
        if args.sesans:
            # make extended q-range for sesans
            aliasses_aggr = ['aggregation','aggr','aggregate','frac2d']
            if stype in aliasses_aggr:
                Reff,Naggr,fracs_aggr = S_par
                qmin = 0.001 * np.pi/(2*Reff)
                deltamax = 3*Reff
            else:
                qmin = 0.001 * np.pi/dmax
                deltamax = 3 * dmax
            qmax = 1e4 * qmin
            qpoints = 5000
            q = np.linspace(qmin,qmax,qpoints)
            delta = np.linspace(0, deltamax, args.deltapoints)
        else:
            q = np.linspace(args.qmin,args.qmax,args.qpoints)

        printt("\n    Calculating intensity, I(q)...")

        ### calculate form factor and forward scattering I0
        I0, Pq = calc_Pq_func(q, r, pr_norm, conc, point_distribution.volume_total)

        ### calculate structure factor
        S = calc_S_func(q,point_distribution, stype, S_par, Pq)
        save_S_func(q,S,model_filename)
        S_list.append(S)

        ### calculate theoretical scattering
        I = calc_Iq_func(q, Pq, S, sigma_r)
        save_I_func(q,I,model_filename)
        I_list.append(I)

        ### simulate SAXS data
        I_sim,sigma = simulate_data_func(q,I,I0,args.exposure)
        save_Isim_func(q,I_sim,sigma,model_filename)
        I_sim_list.append(I_sim)
        sigma_list.append(sigma)

        ### calculate and simulate sesans
        if args.sesans:

            # calculated theoretical SESANS
            G = calc_G_sesans(q,delta,I)

            # simulate sesans data         
            G_sim,sigma_G = simulate_sesans(delta,G,args.sesans_error)
            
            # append to list (in case of multiple models)
            delta_list.append(delta)
            G_list.append(G)
            G_sim_list.append(G_sim)
            sigma_G_list.append(sigma_G)

    printt(" ")
    printt("Generating plots...")
    colors = ['blue','red','green','orange','purple','cyan','magenta','black','grey','pink','forrestgreen']

    #plot 2D projections
    for m in model_filename_list:
        print("    2D projection: points_" + m + ".png ...")
    plot_2D(x_list, y_list, z_list, sld_list, model_filename_list, args.high_res, colors)
    
    #3D vizualization: generate pdb file with points
    for m in model_filename_list:
        print("    3D models: " + m + ".pdb ...")
    generate_pdb(x_list, y_list, z_list, sld_list, model_filename_list)
    
    #plot p(r) and I(q)
    print("    plot pr and Iq and Isim: plot.png ...")
    plot_results(q, r_list, pr_norm_list, I_list, I_sim_list, 
                 sigma_list, S_list, model_name_list, args.xscale_lin, args.high_res, colors)

    #plot and save sesans
    if args.sesans:
        plot_sesans(delta_list, G_list, G_sim_list, sigma_G_list, model_name_list, args.high_res, colors)
        save_sesans(delta_list, G_list, G_sim_list, sigma_G_list, model_filename_list)

    time_total = time.time() - start_total
    printt(" ")
    printt("Simulation successfully completed.")
    printt("    Total run time: " + str(round(time_total, 1)) + " seconds.")
    printt(" ")

    # close log file and copy into model directories
    #f_out.close()
    for model_filename in model_filename_list:
        shutil.copy('shape2sas.log','%s/%s.log' % (model_filename,model_filename))
        if args.high_res:
            shutil.copy('plot.pdf','%s/plot_%s.pdf' % (model_filename,model_filename))
        else:
            shutil.copy('plot.png','%s/plot_%s.png' % (model_filename,model_filename))
        if args.sesans:
            if args.high_res:
                shutil.copy('sesans.pdf','%s/sesans_%s.pdf' % (model_filename,model_filename))
            else:
                shutil.copy('sesans.png','%s/sesans_%s.png' % (model_filename,model_filename))
