from collections import OrderedDict


# acquisition function optimizer configuration
acquisition_opt = OrderedDict(
    [
        ('method', 'L-BFGS-B'),
        ('bounds', None),
        ('jac', True),
        ('options',
         {
             'disp': True,
             'ftol': 2.220446049250313e-15,
             'maxiter': 1e02,
         })
    ])

bfgs_opt = OrderedDict(
    [
        ('method', 'BFGS'),
        ('jac', True),
        ('options',
         {
             'disp': True,
             'maxiter': 1e02,
         })
    ])

hyp_opt = OrderedDict(
    [
        ('method', 'L-BFGS-B'),
        ('var_to_bounds', None),
        ('options',
         {
             'disp':False,
             'ftol': 2.220446049250313e-15,
             'maxiter': 1e03,
         })
    ])


KLacquisition_opt = OrderedDict(
    [
        ('method', 'trust-constr'),
        ('jac', True),
        ('constraints', None),
        ('options',
         {
             'disp': True,
             'maxiter': 1e02,
         })
    ])