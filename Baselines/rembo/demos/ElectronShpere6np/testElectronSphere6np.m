function testBranin(varargin)

run_high_dim = 0;
embedding = 1;
embedding_consecutive = 0;
test_percentage = 0;

% total_iter = 500;
% high_dim = 25;
total_iter = 310;
high_dim = 12;
rotate = 0;


%% Test the percentage of success
if test_percentage
    dim = 2;                                         % Intrinsic Dimensionality.
    used_dim = 2;                                              % Dimension used.
    
    maximizers = trueMaximizer();
    scale = max(1.5*log(used_dim));
    test_bounds = stardardBounds(dim)*scale;                  % Standard Bounds.    
    
    prct = success_prctg(high_dim, 1000, dim, used_dim, test_bounds, ...
        maximizers);                         % Success Percentage by simulation.
    
    fprintf('Success Percentage by using %d dimensions is approximately: \n%f.\n', ...
        used_dim, prct);
end


%% Random embedding with true intrisic dimensions
%  True maximum is ensured to fall in to the bounds
if embedding
%     dim = 6;    % embedding dimension
    for dim=[6, 4, 3, 2]
        Xopt = zeros(20, total_iter, high_dim);
        yopt = zeros(total_iter, 20);
        for experiment=1:20
            rng(experiment)
            [model, Xopt_i] = rembo(total_iter, dim, high_dim, rotate, 1, 1, experiment);
            Xopt(experiment, :, :) = Xopt_i;
            yopt(:, experiment) = model.f;
        end
        string_out = sprintf('ElectronSphere6np_d=%d_lcb', dim);
        save(string_out, 'Xopt','yopt');
    end
    minimizer = [[0.48682843, 0.78674212, 0.57885328, 0.31341441, 0.29749929, 0.40868617, 0.07889095, 0.68647393, 0.98678081, 0.2134249, 0.79744743, 0.59130902]];
    ditance_log_plot(total_iter, model.f);
end


%% Random embedding with true intrisic dimensions
%  True maximum is NOT ensured to fall in to the bounds
%  Runs are repeated to ensure covering of the true maximum
if embedding_consecutive
    dim = 2;
    rotate = 0;
    run_iter = 125; num_trial = 4;
    f_values = zeros(run_iter*num_trial,1);
    
    for i = 1:num_trial
        model = rembo(run_iter, dim, high_dim, rotate);
        f_values((i-1)*run_iter+1:i*run_iter) = model.f;    
    end    

    ditance_log_plot(run_iter*num_trial, f_values);
end


%% High Dim
if run_high_dim
    rotate = 0;
    model = rembo(total_iter, high_dim, high_dim, rotate, 0, 0);
    
    ditance_log_plot(total_iter, model.f);
end


%% Run rembo.
    function [model, Xopt_i] = rembo(total_iter, dim, high_dim, rotate, force_in_bounds, ...
        embed, seed)
        % total_iter: total number of iterations.
        % dim: embedding dimension.
        % high_dim: ambient dimension.
        % roate: whether to randomly rotate the objective function.
        % force_in_bounds: to force an optimizer in bound by repeatly drawing
        %                  random embedding matrices.
        % embed: Whether to use a randome embedding matrix. (If not 
        %        then we effectively use regular BO.)

        
    
        if nargin < 5
            force_in_bounds = 0;       % Whether to force an optimizer in bound.
        end

        if nargin < 6
            embed = 1;                        % Whether to use embedding or not.
        end

        if rotate
            % Rotate the objective function.
            [rm, ~] = qr(randn(high_dim, high_dim), 0);
        else
            % Do not rotate the objective function.
            rm = eye(high_dim);
        end

        if embed
            % Generate random projection matrix A.
%             A_rand = randn(high_dim, dim);
%             A = orth(A_rand);
            % Load random projection matrix A 12x12
            A12x12 = 0;
            string_A = sprintf('A_orth_matrices/A_orth_seed=%d', seed);
            load(string_A);
            A = A12x12(1:high_dim, 1:dim);
%             [in_bounds, A] = test_fall_in_bound(dim, rm);       
%             while ~in_bounds && force_in_bounds
%                 % Ensure that at least one maximizer fall in bound by 
%                 % generating as many random projection matrix A as needed.
%                 [in_bounds, A] = test_fall_in_bound(dim, rm);
%             end
        else
            % By setting A to be identity we do not use embedding here.
            A = eye(high_dim, dim);
        end

%         scale = max(1.5*log(dim), 1);
        scale = 1/sqrt(dim);
        bounds = stardardBounds(dim)*scale;                 % Initialize bounds.
%         obj_fct = @(x)-branin((A*x')');     % Initialize the objective function.
%         obj_fct = @(x)-ElectronSphere6np((A*x')');
%         obj_fct = @(x)-ElectronSphere6np(max(min((A*x')', 1), 0));
        obj_fct = @(x)-ElectronSphere6np(max(min((((A*x')+1).* 0.5)', 1), 0));

%         init_pt = zeros(1, dim);                                % Initial point.
%         init_pt = rand(1, dim);
%         init_pt = (pinv(A)*rand(10, high_dim)')';
        init_pt10 = rand(10, high_dim);
%         init_f = obj_fct(init_pt);                     % Evaluate initial point.
        init_f_high_dim = -ElectronSphere6np(init_pt10);
        [init_f, index_max] = max(init_f_high_dim);
        init_pt_high_dim = init_pt10(index_max, :);
        init_pt = (pinv(A)*init_pt_high_dim')';

        hyp = [ones(dim, 1)*1 ; 1];          % Setup initial hyper-parameters.
        hyp = log(hyp);

        % Initialize model.
        model = init_model(dim, bounds, init_pt, init_f, hyp, 1e-6, 'M52ard');
        % Do optimization.
        model = sparse_opt(obj_fct, total_iter-1, model);
%         Xopt_i = (A*model.X(1:model.n, :)')';
%         Xopt_i = max(min((A*model.X(1:model.n, :)')', 1), 0);
%         Xopt_i = max(min((A*model.X(11:model.n, :)')', 1), 0);
        Xopt_i = max(min(((A * model.X(11:model.n, :)')' + 1).* 0.5, 1), 0);
%         fopt_i = -ElectronSphere6np(Xopt_i);
%         err_fs = abs(fopt_i - model.f(11:model.n));
        Xopt_i = [init_pt10; Xopt_i];
    end

%% Helper functions.
    function ditance_log_plot(total_iter, fvalues)
        maximazer = trueMaximizer();
        figure;
        dis = zeros(total_iter,1);
        for i =1:total_iter
            dis(i) = -branin(maximazer(:, 1)') - max(fvalues(1:i));
        end
        loglog(1:total_iter, dis); 
    end

    function [in_bounds, A] = test_fall_in_bound(used_dim, rm)
        maximizers = trueMaximizer();
        test_bounds = stardardBounds(2);
        scale = max(1.5*log(used_dim));
        test_bounds = test_bounds*scale;

        [prct, A] = success_prctg(high_dim, 1, 2, used_dim, test_bounds,...
            maximizers, rm);
        in_bounds =  prct;
        
        if in_bounds
            fprintf('At least one maximizer in bounds.\n');
        else
            fprintf('NO maximizer in bounds.\n');
        end
    end


    function [maximizers] = trueMaximizer() 
        bounds_branin = [-5,10; 0, 15];
        maximizers = [pi, -pi, 9.42478; 2.275, 12.275, 2.475];
        maximizers = bsxfun(@minus, maximizers, bounds_branin(:, 1));
        maximizers = bsxfun(@rdivide, maximizers, bounds_branin(:, 2) - ...
            bounds_branin(:, 1))*2-1;
    end

    function [prct, A] = success_prctg(high_dim, num_trial, dim,...
        used_dim, bounds, maximizers, rm)

        total = 0;
        cmbnts = combntns(1:used_dim,dim);
        num_maximizers = size(maximizers, 2);

        for i = 1:num_trial
            indices = 1:high_dim;
            A = randn(high_dim, used_dim);

            if nargin > 6
                A = rm*A;
            end
            fail = 1;

            for j = 1:size(cmbnts, 1)
                for k = 1:num_maximizers
                    true_maximizer = inv(A(indices(1:dim), cmbnts(j, :))) * ...
                        maximizers(:, k);
                    if ~(sum(true_maximizer <= bounds(:,2)) < dim || ...
                        sum(true_maximizer >= bounds(:,1)) < dim)
                        fail = 0;
                    end
                end
            end
            total = total + fail;
        end
        prct = 1 - total/num_trial;
    end
end
