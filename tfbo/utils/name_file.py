def name_file_start(dict):
    str_add = '_'.join([str(val_i) + str_i for str_i, val_i in dict.items()])
    file_x = 'x_' + str_add
    file_y = 'y_' + str_add
    return file_x, file_y

def name_synthetic_dict(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim', 'quantile']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_dict_wo_quantile(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Sensitivity'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_synthetic_dict_no_quantile(dict):
    keys = ['obj', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_synthetic(dict):
    keys = ['obj', 'opt', 'loss', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

def name_model_vae(dict):
    keys = ['obj', 'opt', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file