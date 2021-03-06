function [ y ] = ElectronSphere6np( x )
%ELECTRONSPHERE6NP Summary of this function goes here
%   x is assumed to be N x high_dim
n_p = 6;
high_dim = n_p.*2;

[N, shape1] = size(x);
x_reshape = permute(reshape(x, [N, 2, n_p]), [1, 3, 2]);

theta = x_reshape(:, :, 1) .* 2 .* pi;
phi = x_reshape(:, :, 2) .* pi;


spherical = zeros(N, n_p, 2);
spherical(:, :, 1) = theta;
spherical(:, :, 2) = phi;
[x_pt, y_pt, z_pt] = spherical_to_cartesian(spherical);

x_Mat = zeros(N, n_p, n_p);
y_Mat = zeros(N, n_p, n_p);
z_Mat = zeros(N, n_p, n_p);
for k=1:N
    for i=1:n_p
        for j=1:n_p
            x_Mat(k, i, j) = x_pt(k, i) - x_pt(k, j);
            y_Mat(k, i, j) = y_pt(k, i) - y_pt(k, j);
            z_Mat(k, i, j) = z_pt(k, i) - z_pt(k, j);
        end
    end
end
x_Mat2 = x_Mat .^ 2;
y_Mat2 = y_Mat .^ 2;
z_Mat2 = z_Mat .^ 2;

Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) .^ (- 0.5);
sum_fx = zeros(N, 1);
for l=1:N
    sum_fx(l, :) = sum(sum(triu(squeeze(Mat2(l, :, :)),1)));
end

if max(sum_fx) == inf
    sum_fx(sum_fx == ones(N, 1)*inf, :) = 1e09;
end

y = sum_fx + normrnd(0., 0.01, [N, 1]);
% y = sum_fx;
end

function [ x_pt, y_pt, z_pt ] = spherical_to_cartesian( alpha_reshape )
x_pt = cos(alpha_reshape(:, :, 1)) .* sin(alpha_reshape(:, :, 2));% N x n_p
y_pt = sin(alpha_reshape(:, :, 1)) .* sin(alpha_reshape(:, :, 2));% N x n_p
z_pt = cos(alpha_reshape(:, :, 2));                               % N x n_p
end