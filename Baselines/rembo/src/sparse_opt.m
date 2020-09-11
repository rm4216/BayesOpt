function model = sparse_opt(objective_func, num_iter, model, loss, special_fct)

dopt.maxevals = 500;
dopt.maxits = 200;
dopt.showits = 0;

for i = 1:num_iter
    if strcmp(loss,'lcb')
        final_xatmin = maximize_acq(model, dopt, 'ucb');
    elseif strcmp(loss,'Neg_ei')
        final_xatmin = maximize_acq(model, dopt, 'ei');
    elseif strcmp(loss,'Neg_pi')
        final_xatmin = maximize_acq(model, dopt, 'pi');
    end

    if model.high_dim > 0
        [f_t, record] = objective_func(final_xatmin');
        model.records(i+1, :) = record;
    else
        f_t = objective_func(final_xatmin');
        record = 0;
    end
    
    model = update_model(model, f_t, final_xatmin');

    if nargin > 4 
        special_fct(model);
    end
    
end


end
