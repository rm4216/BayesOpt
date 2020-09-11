loss = {'lcb','Neg_ei','Neg_pi'};
path_load = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronShpere6np/scp/';
path_save = '/home/rm4216/Desktop/ImperialCollege/Python/Github_manifold_bo/BayesOpt/Baselines/rembo/demos/ElectronShpere6np/';
total_iter = 310;
high_dim = 12;
for jj = 1 : length(loss)
    for dim = [6, 4, 3, 2]
        Xopt_all = zeros(20, total_iter, high_dim);
        yopt_all = zeros(total_iter, 20);
        for seed = 1:20
            string_load = [path_load,'ElectronSphere6np_',loss{jj},'_d=',num2str(dim),'_seed_',num2str(seed),'.mat'];
            load(string_load)
            Xopt_all(seed, :, :) = Xopt;
            yopt_all(:, seed) = yopt;
        end
        Xopt = Xopt_all;
        yopt = yopt_all;
        string_save = [path_save,'ElectronSphere6np_d=',num2str(dim),'_',loss{jj}];
        save(string_save, 'Xopt','yopt');
    end
end