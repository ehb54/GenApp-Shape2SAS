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

def check_unique(A_list):
    """
    if all elements in a list are unique then return True, else return False
    """
    unique = True
    N = len(A_list)
    for i in range(N):
        for j in range(N):
            if i != j:
                if A_list[i] == A_list[j]:
                    unique = False

    return unique

if __name__=='__main__':
    
    argv_io_string = io.StringIO(sys.argv[1])
    json_variables = json.load(argv_io_string)
    message = genapp(json_variables)

    ## initialize execute command
    path = os.path.dirname(os.path.realpath(__file__))
#    command_to_execute = [path + '/shape2sas.py']
    command_to_execute = ['/opt/miniconda3/bin/python',path + '/shape2sas.py']

    ## global input (from GUI)
    qmin = float(json_variables['qmin'])
    qmax = float(json_variables['qmax'])
    qpoints = int(json_variables['qpoints']) # number of points in (simulated) q
    exposure = float(json_variables['exposure'])
    prpoints = int(json_variables['prpoints']) # number of points in p(r)
    Npoints = int(json_variables['Npoints']) # max number of points per model

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

    command_to_execute.extend(['--qmin',json_variables['qmin']])
    command_to_execute.extend(['--qmax',json_variables['qmax']])
    command_to_execute.extend(['--qpoints',json_variables['qpoints']])
    command_to_execute.extend(['--exposure',json_variables['exposure']])
    command_to_execute.extend(['--prpoints',json_variables['prpoints']])
    command_to_execute.extend(['--Npoints',json_variables['Npoints']])
    if high_res:
        command_to_execute.extend(['--high_res'])
    
    Model_list,model_name_list = [],[]
    Max_Number_of_Models = 4
    for i in range(Max_Number_of_Models):
        n = i+1

        try:
            dummy = json_variables['include_model_%d' % n]
            model_name = json_variables['name_%d' % n]
            Model = '_%d' % n
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
            command_to_execute.extend(['--subunit_type', ",".join(json_variables['subunit_type_%d' % n])])
            command_to_execute.extend(['--sld'])
            for j in range(len(json_variables['p_%d' % n])):
                command_to_execute.extend([json_variables['p_%d' % n][j]])
            command_to_execute.extend(['--dimension'])
            for j in range(len(json_variables['a_%d' % n])):
                if b in (None, "", 0):
                    command_to_execute.extend([json_variables['a_%d' % n][j]]) 
                elif c in (None, "", 0):
                    command_to_execute.extend([" ".join(json_variables['a_%d' % n][j],json_variables['b_%d' % n][j]])
                else:
                    command_to_execute.extend([" ".join(json_variables['a_%d' % n][j],json_variables['b_%d' % n][j],json_variables['c_%d' % n][j]])
            command_to_execute.extend(['--com', " ".join(f"{x},{y},{z}" for x, y, z in zip(json_variables['x_%d' % n], json_variables['y_%d' % n], json_variables['z_%d' % n]))])
            command_to_execute.extend(['--polydispersity',json_variables['polydispersity_%d' % n]])
            command_to_execute.extend(['--S',json_variables['S_%d' % n]])
            command_to_execute.extend(['--conc',json_variables['conc_%d' % n]])
            if json_variables['S_%d' % n] == 'HS':
                command_to_execute.extend(['--Spar ',json_variables['r_hs_%d' % n],',',json_variables['conc_%d' % n]]) # will this work? is the order correct? - disable Spar for now, for testing... 
            if json_variables['S_%d' % n] == 'Aggr':
                fracs_aggr = float(json_variables['frac_%d' % n]) # fraction of particles in aggregated form
                R_aggr = float(json_variables['R_eff_%d' % n]) # effective radius per particle in aggregate
                N_aggr = float(json_variables['N_aggr_%d' % n]) # number of particles per aggregate
                command_to_execute.extend(['--Spar ',fracs_aggr,',',R_aggr,',',N_aggr]) # will this work? is the order correct?- disable Spar for now, for testing... 
            command_to_execute.extend(['--sigma_r',json_variables['sigma_r_%d' % n]])
#            command_to_execute.extend(['--scale',json_variables['scale%d' % n]])

            Model_list.append(Model)
            model_name_list.append(model_name)
        except:
            pass

    ## run shape2sas
    message.udpmessage({"_textarea": 'running Shape2SAS\n'}) 
#    [message.udpmessage({"_textarea": cmd + "\n"}) for cmd in command_to_execute]
    f = open('stdout.dat','w')
    execute(command_to_execute,f)
    f.close()

    ## Retrieve Dmax and Rg
    dmax = 1.1111
    Rg = 2.222

    ## send output to GUI
    output = {} # create an empty python dictionary
    folder = json_variables['_base_directory'] # output folder dir
    output["fig"] = "%s/plot.png" % folder
    
    # model-dependent output
    unique = check_unique(Model_list)
    for (Model,model_name) in zip(Model_list,model_name_list):
        model_name2 = model_name.replace(" ", "_")
        if unique:
            filename = "%s.zip" % model_name2
        else:
            filename = "%s_model%s" % (model_name2,Model)
        output["points%s" % Model] = "%s/points_%s.png" % (folder,model_name2)
        output["pdb_jmol%s" % Model] = "%s/%s.pdb" % (folder,model_name2)
        output["pdb%s" % Model] = "%s/%s.pdb" % (folder,model_name2)
        output["pr%s" % Model] = "%s/pr_%s.dat" % (folder,model_name2)
        output["Iq%s" % Model] = "%s/Iq_%s.dat" % (folder,model_name2)
        output["Isim%s" % Model] = "%s/Isim_%s.dat" % (folder,model_name2)
        os.system('zip %s pr%s.dat Iq_%s.dat Sq_%s.dat Isim_%s.dat %s.pdb points_%s.png plot.png' % (filename,model_name2,model_name2,model_name2,model_name2,model_name2,model_name2))
        output["zip%s" % Model] = "%s/%s" % (folder,filename)
        output["Dmax%s" % Model] = "%1.2f" % dmax
        output["Rg%s" % Model] = "%1.2f" % Rg
    
    print( json.dumps(output) ) # convert dictionary to json and output

