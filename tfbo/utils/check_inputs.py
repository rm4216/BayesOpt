def verify_dict_inputs(dict):
    for key_i, val_i in dict.items():
        if val_i is None:
            raise AttributeError('missing required field: --' + key_i)

def transform_inputs(string):
    string_list = string.split('_')
    def list_numbers(string_in):
        return int(string_in)

    list_num = list(map(list_numbers, string_list))
    return list_num