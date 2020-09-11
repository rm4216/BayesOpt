import numpy as np
from collections import OrderedDict


def dict_outputs(list_tuples, dict_out, key_bl):
    '''
    structure assumed for each element in a list: {(x_i, y_i, hyp_i)}_{i=1}^{n_initializations}
    :param list_tuples:
    :return:
    '''
    x_out = np.stack([tup_i[0] for tup_i in list_tuples], axis=0)
    y_out = np.concatenate([tup_i[1] for tup_i in list_tuples], axis=1)     # check shape
    hyp_out = np.stack([tup_i[2] for tup_i in list_tuples], axis=0)
    log_lik_out = np.stack([tup_i[3] for tup_i in list_tuples], axis=0)

    if dict_out[key_bl] is not None:
        xprev = dict_out[key_bl]['Xepisodes']
        yprev = dict_out[key_bl]['Yepisodes']
        hyp_prev = dict_out[key_bl]['hyp_episodes']
        log_lik_prev = dict_out[key_bl]['log_lik_episodes']

        x_out = np.concatenate([xprev, x_out], axis=0)
        y_out = np.concatenate([yprev, y_out], axis=1)
        hyp_out = np.concatenate([hyp_prev, hyp_out], axis=0)
        log_lik_out = np.concatenate([log_lik_prev, log_lik_out], axis=0)

    dict_out[key_bl] = OrderedDict(
        [
            ('Xepisodes', x_out),
            ('Yepisodes', y_out),
            ('hyp_episodes', hyp_out),
            ('log_lik_episodes', log_lik_out)
        ])
    return dict_out

def attach_dictionaries(dict_a, dict_b):
    for key_i in dict_a:
        # check contains filed: dict = { Xepisodes: x, Yepisodes: y, hyp_episodes: hyp}
        if dict_a[key_i] is not None and dict_b[key_i] is not None:
            dict_a[key_i] = attach_subdictionaries(dict_a[key_i], dict_b[key_i])
    return dict_a

def initialize_dictionary():
    dict_init = OrderedDict(
                [
                    ('add_bo', None),
                    ('DiagNN_bo', None),
                    ('DiagNNKL_bo', None),
                    ('FullNN_bo', None),
                    ('FullNNKL_bo', None),
                    ('rembo', None),
                    ('vae_bo', None)
                ])
    # dict_init = OrderedDict(
    #             [
    #                 ('dgp_bo', None),
    #                 ('NN_bo', None),
    #                 ('dgpMo_bo', None),
    #             ])
    return dict_init

def attach_subdictionaries(subdict_a, subdict_b):
    # iterate over the fields Xepisodes, Yepisodes, hyp_episodes
    # all fields need to contain a value
    for key_i in subdict_a:
        assert subdict_a[key_i] is not None and subdict_b[key_i] is not None
        subdict_a[key_i] = concatenate_values(subdict_a[key_i], subdict_b[key_i], key_i)
    return subdict_a

def concatenate_values(input_a, input_b, key):
    # assign different axis to Yepisodes
    if key == 'Yepisodes':
        N = np.min([input_a.shape[0], input_b.shape[0]])
        return np.concatenate([np.copy(input_a[0:N, :]), np.copy(input_b[0:N, :])], axis=1)
    else:
        N = np.min([input_a.shape[1], input_b.shape[1]])
        return np.concatenate([np.copy(input_a[:, 0:N, :]), np.copy(input_b[:, 0:N, :])], axis=0)