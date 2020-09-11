"""

This version of autoencoder is able to save weights and load weights for the
encoder and decoder portions of the network

"""

# from gpu_utils import pick_gpu_lowest_memory
# gpu_free_number = str(pick_gpu_lowest_memory())
#
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_free_number)

import argparse
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
import yaml
import time
import os
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop

import sys,os
sys.path.insert(0, os.path.join(sys.path[0], '..', '..'))
from Baselines.chemvae import hyperparameters
from Baselines.chemvae import mol_utils as mu
from Baselines.chemvae import mol_callbacks as mol_cb
from keras.callbacks import CSVLogger

from Baselines.chemvae.models import encoder_model, load_encoder
from Baselines.chemvae.models import decoder_model, load_decoder
from Baselines.chemvae.models import property_predictor_model, load_property_predictor
from Baselines.chemvae.models import variational_layers
from functools import partial
from keras.layers import Lambda
from collections import OrderedDict


def load_models(params):

    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params['kl_loss_weight'])

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # # Decoder
    # if params['do_tgru']:
    #     x_out = decoder([z_samp, x_in])
    # else:
    #     x_out = decoder(z_samp)
    x_out = decoder(z_samp)

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ) and
                ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 )):

            reg_prop_pred, logit_prop_pred   = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.extend([reg_prop_pred,  logit_prop_pred])

        # regression only scenario
        elif ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name='logit_prop_pred')(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError('no logit tasks or regression tasks specified for property prediction')

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print('x_mean shape in kl_loss: ', x_mean.get_shape())
    kl_loss = - 0.5 * \
        K.mean(1 + x_log_var - K.square(x_mean) -
              K.exp(x_log_var), axis=-1)
    return kl_loss


def main_property_run(params, X_train, Y_train):
    # X_train = X_train[:, :, None]
    start_time = time.time()

    # load data
    # X_train, X_test, Y_train, Y_test = vectorize_data(params)

    # load full models:
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    # model_test_targets = {'x_pred':X_test,
    #     'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    model_loss_weights = {
                    'x_pred': ae_loss_weight*xent_loss_weight,
                    'z_mean_log_var':   ae_loss_weight*kl_loss_var}

    prop_pred_loss_weight = params['prop_pred_loss_weight']


    if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
        model_train_targets['reg_prop_pred'] = Y_train[0]
        # model_test_targets['reg_prop_pred'] = Y_test[0]
        model_losses['reg_prop_pred'] = params['reg_prop_pred_loss']
        model_loss_weights['reg_prop_pred'] = prop_pred_loss_weight
    if ('logit_prop_tasks' in params) and (len(params['logit_prop_tasks']) > 0 ):
        if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0 ):
            model_train_targets['logit_prop_pred'] = Y_train[1]
            # model_test_targets['logit_prop_pred'] = Y_test[1]
        else:
            model_train_targets['logit_prop_pred'] = Y_train[0]
            # model_test_targets['logit_prop_pred'] = Y_test[0]
        model_losses['logit_prop_pred'] = params['logit_prop_pred_loss']
        model_loss_weights['logit_prop_pred'] = prop_pred_loss_weight


    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)

    callbacks = [ vae_anneal_callback, csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    # control verbose output
    keras_verbose = params['verbose_print']

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                params=params, prop_pred_model = property_predictor,save_best_only=False))

    AE_PP_model.compile(loss=model_losses,
               loss_weights=model_loss_weights,
               optimizer=optim,
               metrics={'x_pred': ['categorical_accuracy',
                    vae_anneal_metric]})

    loss_PP_0 = AE_PP_model.evaluate(x=X_train[:, :, None], y=model_train_targets, batch_size=params['batch_size'], verbose=1,
                                     sample_weight=None, steps=None)
    output0 = AE_PP_model.predict(X_train[:, :, None], batch_size=params['batch_size'])
    X_dec0 = output0[0].copy()
    z_mean0, _ = encoder.predict(X_train[:, :, None])
    X_dec_z0 = decoder.predict(z_mean0)
    err_dec = np.max(np.abs(X_dec_z0 - X_dec0))

    AE_PP_model.fit(X_train[:, :, None], model_train_targets,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         initial_epoch=params['prev_epochs'],
                         callbacks=callbacks,
                         verbose=keras_verbose)
         # validation_data=[X_test, model_test_targets]
     # )
    loss_PP_1 = AE_PP_model.evaluate(x=X_train[:, :, None], y=model_train_targets, batch_size=params['batch_size'], verbose=1,
                                     sample_weight=None, steps=None)
    output1 = AE_PP_model.predict(X_train[:, :, None], batch_size=params['batch_size'])
    X_dec1 = output1[0].copy()
    z_mean1, _ = encoder.predict(X_train[:, :, None])
    X_dec_z1 = decoder.predict(z_mean1)
    err_dec1 = np.max(np.abs(X_dec_z1 - X_dec1))

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    property_predictor.save(params['prop_pred_weights_file'])

    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')

    return AE_PP_model, encoder, decoder, property_predictor, kl_loss_var


# if __name__ == "__main__":

def init_train(Xtrain=None, Ytrain=None):

    names = ['seed', 'obj', 'opt', 'loss', 'proj_dim', 'input_dim']
    defaults = ['0', 'RosenbrockLinear10D', 'vae_bo', 'Neg_pi', int(10), int(60)]
    types = [str, str, str, str, int, int]
    helps = ['seed_number', 'objective_name', 'optimizer_name', 'loss_name', 'proj_dim', 'input_dim']

    parser = argparse.ArgumentParser(
        description='Input: seed_number, objective_name, optimizer_name, loss_name, proj_dim,'
                    ' input_dim, maxiter')
    for name_i, default_i, type_i, help_i in zip(names, defaults, types, helps):
        parser.add_argument('--' + name_i, default=default_i, type=type_i, help=help_i)
    args_p = parser.parse_args()
    args = vars(args_p)

    # args = OrderedDict(
    #     [
    #         ('seed', int(0)),
    #         ('obj', 'MichalewiczLinear10D'),
    #         ('opt', 'vae_bo'),
    #         ('loss', 'Neg_pi'),
    #         ('proj_dim', int(10)),
    #         ('input_dim', int(60))
    #     ])

    path = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/chemvae/settings/'
    filename = name_model_vae(args)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--exp_file',
    #                     help='experiment file', default='exp.json')
    # # parser.add_argument('-d', '--directory',
    # #                     help='exp directory', default=None)
    # parser.add_argument('-d', '--directory',
    #                     help='exp directory', default='/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/chemical_vae-master/models/zinc_properties/')
    # args = vars(parser.parse_args())
    # if args['directory'] is not None:
    #     args['exp_file'] = os.path.join(args['directory'], args['exp_file'])

    params = hyperparameters.load_params(path + filename + '.json')
    print("All params:", params)

    np.random.seed(123)
    Xtrain = np.random.uniform(low=0., high=1., size=[int(500), args['input_dim']])
    from tfbo.utils.import_modules import import_attr
    task_attr = import_attr('datasets/tasks/all_tasks', attribute=args['obj'])
    objective = task_attr()
    Ytrain = [objective.f(Xtrain, fulldim=False, noisy=True)]

    # X_mean = np.mean(Xtrain, axis=0, keepdims=True)
    # X_std = np.std(Xtrain, axis=0, keepdims=True)

    Y_mean = np.mean(Ytrain[0].copy(), axis=0, keepdims=True)
    Y_std = np.std(Ytrain[0].copy(), axis=0, keepdims=True)

    # Xnorm = (Xtrain - X_mean) / X_std
    from scipy.stats import norm
    Xprobit = norm.ppf(Xtrain)
    Ynorm = (Ytrain[0].copy() - Y_mean) / Y_std  # check all shapes


    AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = main_property_run(params, Xprobit, [Ynorm])
    output = AE_PP_model.predict(Xprobit[:, :, None], batch_size=params['batch_size'])

    return

def name_model_vae(dict):
    keys = ['obj', 'opt', 'proj_dim', 'input_dim']
    name_file = 'Synthetic'
    for key_i in keys:
        name_file += '_' + key_i + str(dict[key_i])
    return name_file

aa = init_train()