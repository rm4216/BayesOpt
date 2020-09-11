ratios = zeros(20, 1);
for experiment = 1:20
    rng(experiment)

    dim = 6;
    N = 1000;
    Xmc = rand(N, dim);      % samples in [0, 1] interval

    scale = 5/sqrt(dim);
    ub = 5/sqrt(dim);
    lb = -5/sqrt(dim);

    Xmc_bounds = (Xmc .* (ub - lb)) + lb;      % samples in [lb, ub]
    Xmc_m11 = (Xmc .* 2) - 1;
    Xmc_bounds2 = Xmc_m11 .* scale;

    error_Xmc_bounds = max(max(abs(Xmc_bounds - Xmc_bounds2)));

    path_load = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronShpere6np/A_orth_matrices/';
    string_load = [path_load,'A_orth_seed=',num2str(experiment),'.mat'];
    load(string_load)

    high_dim = 12;
    A = A12x12(1:high_dim, 1:dim);

    Xmc_clipped = max(min((((A * Xmc_bounds2') + 1).* 0.5)', 1), 0);

    indices = any(Xmc_clipped == 1, 2);
    X_clipped_1s = Xmc_clipped(indices, :);
    index = or(any(Xmc_clipped == 1, 2), any(Xmc_clipped == 0, 2));
    X_clipped_1s0s = Xmc_clipped(index, :);

    ratios(experiment, :) = sum(index)/N;
end
mean_ratios = mean(ratios);
aaa = 5;