#!/opt/miniconda3/bin/python

import json
import io
import sys
import os
from genapp3 import genapp
import numpy as np
import time
import subprocess

def execute(command,f):
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    d = genapp(json_variables)
    start_time = time.time()
    maximum_output_size = 1000000 # maximum output size in number of characters
    maximum_time = 300
    total_output_size = 0
    popen = subprocess.Popen(command, stdout=subprocess.PIPE,bufsize=1)
    lines_iterator = iter(popen.stdout.readline, b"")
    while popen.poll() is None:
        for line in lines_iterator:
            nline = line.rstrip()
            nline_latin = nline.decode('latin')
            total_output_size += len(nline_latin)
            total_time = time.time() - start_time
            if total_output_size > maximum_output_size:
                popen.terminate()
                out_line = '\n\n!!!ERROR!!!\nProcess stopped - could not find solution. Is data input a SAXS/SANS dataset with format (q,I,sigma)?\n\n'
                message.udpmessage({"_textarea": out_line})
                sys.exit()
            elif total_time > maximum_time:
                popen.terminate()
                out_line = '\n\n!!!ERROR!!!\nProcess stopped - reached max time of 5 min (300 sec). Is data input a SAXS/SANS dataset with format (q,I,sigma)?. If data is large (several thousand data points), consider rebinning the data.\n\n'
                message.udpmessage({"_textarea": out_line})
                sys.exit()
            else:
                out_line = '%s\n' % nline_latin
                message.udpmessage({"_textarea": out_line})
            f.write(out_line)
    return out_line

if __name__=='__main__':
    
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    message = genapp(json_variables)

    ## initialize execute command
    path = os.path.dirname(os.path.realpath(__file__))
#    command_to_execute = [path + '/shape2sas.py']
    command_to_execute = ['/opt/miniconda3/bin/python',path + '/shape2sas.py']

    ## plot options
    try:
        dummy = json_variables['xscale_lin']
        xscale_log = False
    except:
        xscale_log = True
    try:
        dummy = json_variables['high_res']
        high_res = True
    except:
        high_res = False


    ## Model-dependent input 
    Model_list,model_name_list = [],[]
    Max_Number_of_Models = 4
    for i in range(Max_Number_of_Models):
        n = i+1
        try:
            dummy = json_variables['include_model_%d' % n]
            Model = '_%d' % n
            model_name = json_variables['name_%d' % n]
            if model_name in model_name_list:
                #model names should be unique
                model_name += Model 
            n_subunit = len(json_variables['subunit_type_%d' % n])
            command_to_execute.extend(['--subunit', ",".join(json_variables['subunit_type_%d' % n])])
            a,b,c = json_variables['a_%d' % n],json_variables['b_%d' % n],json_variables['c_%d' % n]
            command_to_execute.extend(['--dimension'])
            for j in range(n_subunit):
                if b[j] in (None, "", 0):
                    command_to_execute.extend([a[j]])
                elif c[j] in (None, "", 0):
                    command_to_execute.extend([a[j] + ',' + b[j]])
                else:
                    command_to_execute.extend([a[j] + ',' + b[j] + ',' + c[j]])            
            command_to_execute.extend(['--sld'])
            sld = json_variables['sld_%d' % n]
            for j in range(n_subunit):
                command_to_execute.extend([sld[j]])
            command_to_execute.extend(['--com'])
            x,y,z = json_variables['x_%d' % n], json_variables['y_%d' % n], json_variables['z_%d' % n]
            for j in range(n_subunit):
                if x[j][0] == '-':
                    command_to_execute.extend([x[j] + ', ' + y[j] + ', ' + z[j]]) # add space: https://github.com/andreashlarsen/Shape2SAS/issues/4
                else:
                    command_to_execute.extend([x[j] + ',' + y[j] + ',' + z[j]])
            command_to_execute.extend(['--rotation'])
            rot_x,rot_y,rot_z = json_variables['rot_x_%d' % n], json_variables['rot_y_%d' % n], json_variables['rot_z_%d' % n]
            for j in range(n_subunit):
                if rot_x[j][0] == '-':
                    command_to_execute.extend([rot_x[j] + ', ' + rot_y[j] + ', ' + rot_z[j]]) # add space: https://github.com/andreashlarsen/Shape2SAS/issues/4
                else:
                    command_to_execute.extend([rot_x[j] + ',' + rot_y[j] + ',' + rot_z[j]])
            command_to_execute.extend(['--polydispersity',json_variables['polydispersity_%d' % n]])
            try:
                dummy = json_variables['exclude_overlap_%d' % n]
                exclude_overlap = 1
            except:
                exclude_overlap = 0
            command_to_execute.extend(['--model_name',model_name])
            if exclude_overlap:
                command_to_execute.extend(['--exclude_overlap','True'])
            else:
                command_to_execute.extend(['--exclude_overlap','False'])
            command_to_execute.extend(['--conc',json_variables['conc_%d' % n]])
            command_to_execute.extend(['--S',json_variables['S_%d' % n]])
            if json_variables['S_%d' % n] in ['HS','Aggr']:
                command_to_execute.extend(['--S_par'])
            if json_variables['S_%d' % n] == 'HS':
                command_to_execute.extend([json_variables['conc_%d' % n] + ',' + json_variables['r_hs_%d' % n]]) 
            if json_variables['S_%d' % n] == 'Aggr':
                command_to_execute.extend([json_variables['R_eff_%d' % n] + ','  + json_variables['N_aggr_%d' % n] + ',' + json_variables['frac_%d' % n]]) 
            command_to_execute.extend(['--sigma_r',json_variables['sigma_r_%d' % n]])

            Model_list.append(Model)
            model_name_list.append(model_name)
        except:
            pass

    ## global options
    command_to_execute.extend(['--qmin',json_variables['qmin']])
    command_to_execute.extend(['--qmax',json_variables['qmax']])
    command_to_execute.extend(['--qpoints',json_variables['qpoints']])
    command_to_execute.extend(['--exposure',json_variables['exposure']])
    command_to_execute.extend(['--prpoints',json_variables['prpoints']])
    command_to_execute.extend(['--Npoints',json_variables['Npoints']])
    try:
        dummy = json_variables['sesans']
        command_to_execute.extend(['--sesans'])
        sesans = True
    except:
        sesans = False
    ## plot options
    try:
        dummy = json_variables['xscale_lin']
        command_to_execute.extend(['--xscale_lin'])
        xscale_log = False
    except:
        pass
    try:
        dummy = json_variables['high_res']
        command_to_execute.extend(['--high_res'])
    except:
        pass

    ## run shape2sas
    message.udpmessage({"_textarea": 'Running Shape2SAS...\n'}) 
#    [message.udpmessage({"_textarea": cmd + "\n"}) for cmd in command_to_execute]
    f = open('stdout.dat','w')
    execute(command_to_execute,f)
    f.close()

    ## Retrieve Dmax and Rg
    f = open('stdout.dat','r')
    lines = f.readlines()
    f.close()
    dmax_list,Rg_list = [],[]
    for line in lines:
        if 'Rg  :' in line:
            Rg_list.append(float(line.split(':')[1].split('A')[0]))
        if 'Dmax: ' in line:
            dmax_list.append(float(line.split(':')[1].split('A')[0]))

    ## send output to GUI
    output = {} # create an empty python dictionary
    folder = json_variables['_base_directory'] # output folder dir
    output["fig"] = "%s/plot.png" % folder
    if sesans: 
        output["sesans_fig"] = "%s/sesans.png" % folder 
    
    # model-dependent output
    for (Model,model_name,Rg,dmax) in zip(Model_list,model_name_list,Rg_list,dmax_list):
        m = model_name.replace(" ", "_")
        output["pr%s" % Model] = "%s/%s/pr_%s.dat" % (folder,m,m)
        output["Iq%s" % Model] = "%s/%s/Iq_%s.dat" % (folder,m,m)
        output["Isim%s" % Model] = "%s/%s/Isim_%s.dat" % (folder,m,m)
        if sesans:
            output["G%s" % Model] = "%s/%s/G_%s.ses" % (folder,m,m)
            output["Gsim%s" % Model] = "%s/%s/Gsim_%s.ses" % (folder,m,m)
        output["pdb%s" % Model] = "%s/%s/%s.pdb" % (folder,m,m)
        os.system('zip -r %s.zip %s' % (m,m)) 
        output["zip%s" % Model] = "%s/%s.zip" % (folder,m)
        output["Rg%s" % Model] = "%1.2f" % Rg
        output["Dmax%s" % Model] = "%1.2f" % dmax
        output["points%s" % Model] = "%s/%s/points_%s.png" % (folder,m,m)
        output["pdb_jmol%s" % Model] = "%s/%s/%s.pdb" % (folder,m,m)
    
    print( json.dumps(output) ) # convert dictionary to json and output

