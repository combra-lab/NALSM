import pickle as pk
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import sys


def check_create_save_dir(save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(
            'SAVE DIRECTORY ALREADY EXISTS, TERMINATE PROGRAM IMMEDIATELY TO PREVENT LOSS OF EXISTING DATA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def unpack_file(filename, dataPath):

    data_fn = os.path.abspath(os.path.join(dataPath, filename))

    names = []
    data = []

    f = open(data_fn, 'rb')

    read = True
    while read == True:
        dat_temp = pk.load(f)
        if dat_temp == 'end':
            read = False
        else:
            # print(isinstance(dat_temp, str))
            if isinstance(dat_temp, str):
                names.append(dat_temp)
                data.append(pk.load(f))
                # print(data)
    f.close()


    return names, data


def save_tf_data(names, data, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(sess.run(data[i]), f)
    pk.dump('end',f)

    sess.close()
    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)


def save_tf_nontf_data(names, data, names_nontf, data_nontf, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(sess.run(data[i]), f)

    sess.close()

    for i in range(0,len(names_nontf)):
        pk.dump(names_nontf[i],f)
        pk.dump(data_nontf[i], f)

    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)


def save_non_tf_data(names, data, filename, savePath):

    check_create_save_dir(savePath)

    data_fn = os.path.abspath(os.path.join(savePath, filename))

    f = open(data_fn, 'wb')

    for i in range(0,len(names)):
        pk.dump(names[i],f)
        pk.dump(data[i], f)
    pk.dump('end',f)

    f.close()
    print('File__'+str(filename)+'__saved to__'+data_fn)


def save_log_file_of_parameters(root_savename,savepath,parameter_names,parameters_values):

    log_fn = os.path.abspath(os.path.join(savepath, root_savename+'.txt'))

    save_non_tf_data(names=parameter_names, data=parameters_values, filename=root_savename+'.params', savePath=savepath)

    with open(log_fn, 'w') as f:
        for ppp in range(0,len(parameter_names)):
            f.write(str(parameter_names[ppp])+':         '+str(parameters_values[ppp])+'\n')
    print('LOG SAVED.')

















def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def str_splitter_into_2_intervals(str):

    idx_split = str.find('_')
    int1 = int(str[0:idx_split])
    int2 = int(str[idx_split+1:len(str)])

    return [int1,int2]

def func_exp(x, a, b, c,d):
    return a * np.exp(-b * (x-c)) + d

def func_exp_inv(y,a,b,c,d):
    return np.divide(np.log(np.divide(y-d,a)),-b) + c

def func_linear(x,a,b,c):
    return a*(x-c) + b

def func_linear_inv(x,a,b,c):
    return ((x - b)/a)+c


def func_quad(x,a,b,c):
    return a*((x-b)**2) + c

def func_3deg(x,a,b,c,d,e):
    return a * ((x - b) ** 3) + c * ((x - d) ** 2)+ e

def r_square_for_curver_fit(y_data,y_model):
    residuals = y_data - y_model
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

    return 1 - (ss_res / ss_tot)

def residual_sum_of_sqrs(y_data,y_model):
    residuals = y_data - y_model
    ss_res = np.sum(residuals ** 2)

    return ss_res

def remove_files_op(files_found_l):
    rem_idx_raw = input('List idx to remove sep by commas')
    rem_idx = str(rem_idx_raw)

    fls_l = []
    idx_l = []
    search_on = True
    while search_on:
        split_idx = rem_idx.find(',')
        print(split_idx)
        if split_idx == -1:
            idx_l.append(int(rem_idx))
            fls_l.append(files_found_l[int(rem_idx)])
            search_on = False
            break
        idx_to_remove = int(rem_idx[0:split_idx])
        rem_idx = rem_idx[split_idx + 1:len(rem_idx)]
        idx_l.append(idx_to_remove)
        fls_l.append(files_found_l[idx_to_remove])

    print('Will remove these:')
    print(idx_l)
    proceed = input('Proceed [y/n]')
    if proceed == 'n':
        sys.exit(2)

    for kk in range(0, len(fls_l)):
        files_found_l.remove(fls_l[kk])

    return files_found_l,proceed

def select_files_op(files_found_l):

    sel_idx_raw = input('List idx to select sep by commas')
    sel_idx = str(sel_idx_raw)

    fls_l = []
    idx_l = []
    search_on = True
    while search_on:
        split_idx = sel_idx.find(',')
        print(split_idx)
        if split_idx == -1:
            idx_l.append(int(sel_idx))
            fls_l.append(files_found_l[int(sel_idx)])
            search_on = False
            break
        idx_to_select = int(sel_idx[0:split_idx])
        sel_idx = sel_idx[split_idx + 1:len(sel_idx)]
        idx_l.append(idx_to_select)
        fls_l.append(files_found_l[idx_to_select])

    print('Will process these:')
    print(idx_l)
    proceed = input('Proceed [y/n]')
    if proceed == 'n':
        sys.exit(2)

    process_l = []
    for kk in range(0, len(fls_l)):
        process_l.append(fls_l[kk])

    return process_l,proceed




# def compute_f_statistic(complex_model_rss, simple_model_rss,num_samples,)
