import numpy as np
import glob
import os
import pickle
from collections import OrderedDict
from tfbo.utils.import_modules import import_attr
from keras.models import load_model
# from pdb import set_trace as bp


def savefile(filename, array):
    try:
        np.save(filename, array)
    except:
        np.save(filename.replace('home', 'homes'), array)

def loadfile(filename):
    try:
        array = np.load(filename)
    except:
        array = np.load(filename.replace('home', 'homes'))
    return array

def loadMat(filename):
    path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/datasets/tasks/'
    try:
        array = np.load(path + filename)
    except:
        array = np.load(path.replace('home', 'homes') + filename)
    return array

def load_partname(path, partname):
    path_to_os = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt'
    try:
        os.chdir(path_to_os)
        list_names = glob.glob(path + partname + '*')   # requires a os.path?
        array = np.load(list_names[0])
    except:
        os.chdir(path_to_os.replace('home', 'homes'))
        list_names = glob.glob(path + partname + '*')   # requires a os.path?
        array = np.load(list_names[0])
    return array

def load_dictionary(filename):
    try:
        dict_load = pickle.load(open(filename, 'rb'))
    except:
        dict_load = pickle.load(open(filename.replace('home', 'homes'), 'rb'))
    return dict_load

def save_dictionary(filename, dictionary):
    try:
        pickle.dump(dictionary, open(filename, 'wb'))
    except:
        pickle.dump(dictionary, open(filename.replace('home', 'homes'), 'wb'))

# load start
def load_initializations(dict_args, names):
    dict_start = OrderedDict([(key_i, dict_args[key_i]) for key_i in names[:2]])
    name_attr = import_attr('tfbo/utils/name_file', attribute='name_file_start')
    path = 'datasets/data/' + dict_args['obj'] + '/'
    attr_partload = import_attr('tfbo/utils/load_save', attribute='load_partname')
    def load_pair(seed_i):
        dict_start['seed'] = seed_i
        x_file, y_file = name_attr(dict_start)
        y_start = attr_partload(path=path, partname=y_file)
        x_start = attr_partload(path=path, partname=x_file)
        return (x_start, y_start)
    xy_list = list(map(load_pair, dict_args['seed']))
    return xy_list

def load_SRinitializations(dict_args, names):
    dict_start = OrderedDict([(key_i, dict_args[key_i]) for key_i in names[:2]])
    dict_start['n_samples'] = int(50)
    name_attr = import_attr('tfbo/utils/name_file', attribute='name_file_start')
    path = 'datasets/data/' + dict_args['obj'] + '/' + 'part' + str(dict_args['part']) + '/'
    attr_load = import_attr('tfbo/utils/load_save', attribute='loadfile')
    full_path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/' + path
    def load_pair(seed_i):
        dict_start['seed'] = seed_i
        x_file, y_file = name_attr(dict_start)
        y_start = attr_load(filename=full_path + y_file + '_' + 'part' + str(dict_args['part']) + '.npy')
        x_start = attr_load(filename=full_path + x_file + '_' + 'part' + str(dict_args['part']) + '.npy')
        return (x_start, y_start)
    xy_list = list(map(load_pair, dict_args['seed']))
    return xy_list


def load_keras_model(filename):
    try:
        model = load_model(filename)
    except:
        model = load_model(filename.replace('home', 'homes'))
    return model