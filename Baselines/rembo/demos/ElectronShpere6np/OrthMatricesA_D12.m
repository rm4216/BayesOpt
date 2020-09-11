high_dim = 12;

for seed=1:20
    rng(seed)
    A_rand_i = randn(high_dim, high_dim);
    A12x12 = orth(A_rand_i);
    string_out = sprintf('A_orth_matrices/A_orth_seed=%d', seed);
    save(string_out, 'A12x12');
end