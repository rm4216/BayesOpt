import json
from collections import OrderedDict


def name_model_vae(dict):
    keys = ['obj', 'opt', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file


path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/chemvae/settings/'
path_models = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/chemvae/'
dict_args = OrderedDict(
    [
        ('obj', 'MichalewiczNN10D'),
        ('opt', 'vae_bo'),
        ('proj_dim', int(10)),
        ('input_dim', int(60))
    ])
filename = name_model_vae(dict_args)

dict_params = OrderedDict([('name', 'zinc_prop'),
             ('MAX_LEN', dict_args['input_dim']),
             ('data_file', '250k_rndm_zinc_drugs_clean_3.csv'),
             ('char_file', 'zinc.json'),
             ('encoder_weights_file', path_models + dict_args['obj'] + '/' + filename + '_encoder.h5'),
             ('decoder_weights_file', path_models + dict_args['obj'] + '/' + filename + '_decoder.h5'),
             ('prop_pred_weights_file', path_models + dict_args['obj'] + '/' + filename + '_prop_pred.h5'),
             ('reg_prop_tasks', ['func']),
             ('test_idx_file', 'test_idx.npy'),
             ('history_file', 'history.csv'),
             ('checkpoint_path', './'),
             ('do_prop_pred', True),
             ('TRAIN_MODEL', True),
             ('ENC_DEC_TEST', False),
             ('PADDING', 'right'),
             ('RAND_SEED', 42),
             ('epochs', 120),     # 120
             ('limit_data', 5000),
             ('vae_annealer_start', 29),
             ('dropout_rate_mid', 0.08283292970479479),
             ('anneal_sigmod_slope', 0.5106654305791392),
             ('recurrent_dim', 488),
             ('hidden_dim', dict_args['proj_dim']),
             ('tgru_dropout', 0.19617749608323892),
             ('hg_growth_factor', 1.2281884874932403),
             ('middle_layer', 1),
             ('prop_hidden_dim', 67),
             ('batch_size', 20),
             ('prop_pred_depth', 3),
             ('lr', 0.00045619868229310396),
             ('prop_pred_dropout', 0.15694573998898703),
             ('prop_growth_factor', 0.9902834073131418),
             ('momentum', 0.9902764103622574)])
# dict_params = OrderedDict([('name', 'zinc'),
#              ('MAX_LEN', 60),
#              ('data_file', '250k_rndm_zinc_drugs_clean_3.csv'),
#              ('char_file', 'zinc.json'),
#              ('encoder_weights_file', 'zinc_encoder.h5'),
#              ('decoder_weights_file', 'zinc_decoder.h5'),
#              ('test_idx_file', 'test_idx.npy'),
#              ('history_file', 'history.csv'),
#              ('checkpoint_path', './'),
#              ('do_prop_pred', True),
#              ('TRAIN_MODEL', True),
#              ('ENC_DEC_TEST', False),
#              ('PADDING', 'right'),
#              ('RAND_SEED', 42),
#              ('epochs', 1),  # 70
#              ('vae_annealer_start', 29),
#              ('dropout_rate_mid', 0.08283292970479479),
#              ('anneal_sigmod_slope', 0.5106654305791392),
#              ('recurrent_dim', 488),
#              ('batch_size', 126),
#              ('lr', 0.00039192162392520126),
#              ('hidden_dim', 6),
#              ('tgru_dropout', 0.19617749608323892),
#              ('hg_growth_factor', 1.2281884874932403),
#              ('middle_layer', 1),
#              ('momentum', 0.9717090063868801)])

with open(path + filename + '.json', 'w') as outfile:
    json.dump(dict_params, outfile)