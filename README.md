# Bayesian optimization in feature spaces
High-dimensional Bayesian optimization using low-dimensional feature spaces.
https://arxiv.org/abs/1902.10675

In order to scale Bayesian optimization (BO) to high dimensions, we normally make structural assumptions on the decomposition of the objective and/or exploit the intrinsic lower dimensionality of the problem, e.g., by using random projections. The limitation of aforementioned approaches is the assumption of a linear subspace. We could achieve a higher compression rate  with nonlinear projections, but learning these nonlinear embeddings typically requires much data. We propose to learn a low-dimensional feature space jointly with a) the response surface and b) a reconstruction mapping. In particular we model the response surface with a manifold Gaussian process (mGP) (Calandra et al., 2016; Wilson et al., 2016), and the reconstruction mapping with a multi-output Gaussian process with intrinsic coregionalization model (Goovaerts, 1997).


R. Calandra, J. Peters, C. E. Rasmussen, and M. P. Deisenroth. Manifold Gaussian processes for regression. "International Joint Conference on Neural Networks", 2016.

A. G. Wilson, Z. Hu, R. Salakhutdinov, and E. P. Xing. Deep kernel learning. "International Conference on Artificial Intelligence and Statistics", 2016.

P. Goovaerts. Geostatistics for natural resources evaluation. Oxford University Press, 1997.


# Instructions for setup:
1) Install Tensorflow (version 1.13.1)
2) Install the GPflow package from the GPflow directory running the following commands from the main directory:
```
cd GPflow/
python setup.py install
```
3) Install Keras (version 2.2.4 or earlier)
4) Update Scipy package to version 1.2.1 (or 1.2.0)


# Running Experiments
In order to run the experiments you need to run the bayesian_optimization.py file.
You can also define additional arguments as inputs as follows:

1) Select the random initializations. Each random initialization is stored in "datasets/data" folder. For each objective function there are 20 different initializations for index i=**0;...;19**. This filed takes an int as input.<br />
```
--seed=0
```

2) Select the objective function. All the objective functions are defined in the "datasets/tasks/all_tasks.py". There are four possible choices of objective function (**RosenbrockLinear10D; ProductSinesLinear10D; ProductSinesNN10D; ElectronSphere6np**) that correspond to Rosenbrock and Product of Sines functions with linear and nonlinear feature space, respetively. The intrinsic dimensionality of RosenbrockLinear10D, of ProductSinesLinear10D and ProductSinesNN10D is 10. The last objective function concerns the distribution of electrons on a sphere and has intrinsic dimensionality 12. The input to this field is a string.<br />
```
--obj=RosenbrockLinear10D
```

3) Select the optimizer. All the optimizers are defined in the "tfbo/optimizers" folder. Each optimizer corresponds to a different baseline conforming to specific modeling assumptions. There are 9 different choices (**add_bo; FullNN_bo; FullNNKL_bo; DiagNN_bo; DiagNNKL_bo; NN_bo; NNKL_bo; rembo; vae_bo**). The input to this field is a string.<br />
**NN_bo**: (HMGP-BO) This corresponds to the baseline "HMGP-BO" described in the paper.<br />
**NNKL_bo**: (HMGPC-BO) baseline with additional nonlinear constraint in the acquisition function maximization. The nonlinear constraint is based on Lipschitz continuity of the posterior mean of the multi-output GP and avoids mapping inputs in feature space to zero. The input to this field is a string. This corresponds to the baseline "HMGPC-BO" described in the paper.<br />
**FullNN_bo**: (MGP-BO) baseline with full correlation between output dimensions in the multi-output GP. It inverts efficiently the training covariance matrix of the multi-output GP without independence assumption. It also uses tensor algebra for efficient matrix-vector multiplication. This corresponds to the baseline "MGP-BO" described in the paper.<br />
**FullNNKL_bo**: (MGPC-BO) baseline with full correlation between output dimensions in the multi-output GP and additional nonlinear constraint based on Lipschitz continuity. This corresponds to the baseline "MGPC-BO" described in the paper.<br />
**DiagNN_bo**: (DMGP-BO) This corresponds to the baseline "DMGP-BO" described in the paper.<br />
**DiagNNKL_bo**: (DMGPC-BO) This corresponds to the baseline "DMGPC-BO" described in the paper.<br />
**add_bo**: (ADD-BO) K. Kandasamy, J. Schneider, and B. Poczos. High dimensional Bayesian optimisation and bandits via additive models. "International Conference on Machine Learning", 2015.<br />
**rembo**: (Random embeddings) Z. Wang, M. Zoghi, F. Hutter, D. Matheson, and N. de Freitas.  Bayesian Optimization in High Dimensions via Random Embeddings. "IJCAI", 2013.<br />
**vae_bo**: (VAE-BO) R. Gomez-Bombarelli, N. W. Jennifer, D. Duvenaud, J. M.Hernndez-Lobato, B. Snchez-Lengeling, D. Sheberla,J. Aguilera-Iparraguirre, T. D. Hirzel, R. P. Adams, andA. Aspuru-Guzik.  Automatic chemical design usinga data-driven continuous representation of molecules. "ACS Central Science", 2018.<br />
```
--opt=NN_bo
```

4) Select the acquisition function. All acquisition functions are defined in the "tfbo/models/gp_models.py" file. There are 3 different choices of acquisitions (**Neg_ei; lcb; Neg_pi**) that correspond to expected improvement, lower confidence bound, probability of improvement. The input to this field is a string.<br />
```
--loss=Neg_ei
```

5) Select the dimensionality (proj_dim) of projections or feature space. Some of the baselines are based on a partitioning of the input space which depends also on proj_dim. The input to this field is an int. A suggested choice is <br />
```
--proj_dim=10
```

6) Select dimensionality (input_dim) of input space. Some of the baselines are based on a partitioning of the input space which depends also on input_dim. The input to this field is an int. A suggested choice is <br />
```
--input_dim=60
```

7) Select maximum number of Bayesian optimization iterations. The input to this field is an int <br />
```
--maxiter=300
```

Example of running an experiment from main folder:<br />
```
python tfbo/bayesian_optimization.py --seed=0 --obj=RosenbrockLinear10D --opt=FullNNKL_bo --loss=Neg_ei --proj_dim=10 --input_dim=60 --maxiter=300
```


<!---
# Displaying Results
The "bayesian_optimization.py" script will automatically save the results in the "tests/results" directory. In order to display results:
1) Collect all the seeds for each baseline. The script "tests/results/attach_dicts.py" merges all seeds into a single file.<br />
2) Collect all baselines. The script "tests/results/merge_bl.py" collects the outputs of "attach_dicts.py" for each baseline and merges them in a single dictionary.<br />
3) Plot a comparison. The script "tests/results/merge_plot.py" saves a .pdf of a comparison for each acquisition function.<br />
-->
