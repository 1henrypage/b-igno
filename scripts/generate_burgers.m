% Generates viscous Burgers equation datasets for IGNO training.
%
% PDE:  u_t + u*u_x = nu * u_xx,   (x,t) in [-1,1] x (0,1]
%       u(-1, t) = u(1, t) = 0   (Dirichlet BCs)
%       u(x, 0)  = a(x)           (IC sampled from a Gaussian Process)
%
% viscosity: nu = 0.1/pi  (from DGenNO paper, Appendix sec:burger)
%
% Initial conditions a(x) are sampled from a GRF with FNO-style covariance
% kernel on [-1,1] with Dirichlet BCs, via sine series (DGenNO / GaussianRF).
% Basis: sin(k*pi*x), k=1..N_modes — odd functions, zero at x=-1, 0, +1.
% Kernel: var(a_k) = s^2 * (tau^2 + (k*pi)^2)^(-alpha)
%   In-distribution:     tau=7, alpha=2.5, s=70.14  (std~0.38, matches author ref)
%   Out-of-distribution: tau=5, alpha=2.5, s=70.14  (tau_out/tau_in=5/7~0.72; std~0.67)
%
% PDE solved with Chebfun's pde15s (Chebyshev spectral method):
%   - N=636 Chebyshev collocation points (matches DGenNO paper)
%   - Spectral accuracy, exact BC enforcement
%   - Solution evaluated at 128-point uniform output grid via barycentric interp
% Requires Chebfun: https://github.com/chebfun/chebfun
%
% Output .mat files (v7.3/HDF5) contain:
%   u0     - (N, 128)       initial conditions
%   u_sol  - (N, 101, 128)  full space-time solutions
%   x_mesh - (128, 1)       spatial grid on [-1, 1]
%   t_mesh - (101, 1)       temporal grid on [0, 1]
%
% Dimension convention (MATLAB → Python h5py):
%   MATLAB (N, 128)       → h5py reads (128, N)       → .T → (N, 128)
%   MATLAB (N, 101, 128)  → h5py reads (128, 101, N)  → .T → (N, 101, 128)
% No permutation needed; matches _load_data() in src/problems/burgers.py.
%
% Generated files:
%   data/burgers/viscid_train.mat      — 1000 in-dist training samples  (pool[1:1000])
%   data/burgers/viscid_test_in.mat    —  200 in-dist test samples      (pool[1001:1200])
%   data/burgers/viscid_test_out.mat   —  200 out-of-dist test samples  (separate seed)

clear; clc;

%% ===== DEPENDENCIES =====
% Add Chebfun to path (adjust if your Chebfun is elsewhere)
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '..', 'binaries', 'chebfun'));

%% ===== PARAMETERS — edit these =====
nu = 0.1 / pi;   % viscosity (DGenNO paper, line 617)

tau_in  = 7;     % FNO GRF length-scale, in-distribution   (calibrated to match ref std~0.38)
tau_out = 5;     % FNO GRF length-scale, out-of-distribution (tau_out/tau_in=5/7~0.72)

N_train = 1000;
N_test  = 200;

n_mesh  = 128;
n_time  = 101;   % temporal output grid points (t = 0, 0.01, ..., 1.0)
N_modes = 512;
N_cheb  = 636;   % Chebyshev collocation points for PDE solver (matches DGenNO paper)

% Output directory (relative to project root)
output_dir = fullfile(script_dir, '..', 'data', 'burgers');

% Random seeds (fixed for reproducibility)
% In-dist train+test share one seed (generate N_train+N_test, then split)
% so the test set is drawn from the exact same distribution as training.
seed_in       = 1;
seed_test_out = 3;
%% =====================================

fprintf('=== Burgers Dataset Generation ===\n');
fprintf('  nu = %.6f (= 0.1/pi)\n', nu);
fprintf('  tau_in=%g, tau_out=%g, alpha=2.5, s=70.14\n', tau_in, tau_out);
fprintf('  N_train=%d, N_test=%d\n', N_train, N_test);
fprintf('  Solver: Chebfun pde15s, N_cheb=%d\n', N_cheb);
fprintf('  Output: %s\n\n', output_dir);

% Row vectors: MATLAB (1, n) → h5py reads (n, 1), matches reference format.
x_mesh = linspace(-1, 1, n_mesh);   % (1, 128)
t_mesh = linspace(0,  1, n_time);   % (1, 101)

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

tic_total = tic;

N_in = N_train + N_test;
fprintf('Generating in-dist pool     (N=%4d, tau=%d, seed=%d)...\n', N_in, tau_in, seed_in);
[u0_pool, usol_pool] = gen_samples(N_in, tau_in, N_modes, nu, x_mesh, t_mesh, seed_in, N_cheb);

% Split: first N_train → train, last N_test → test_in
u0_train      = u0_pool(1:N_train, :);
usol_train    = usol_pool(1:N_train, :, :);
u0_test_in    = u0_pool(N_train+1:end, :);
usol_test_in  = usol_pool(N_train+1:end, :, :);

fprintf('Generating out-of-dist test (N=%4d, tau=%d, seed=%d)...\n', N_test, tau_out, seed_test_out);
[u0_test_out, usol_test_out] = gen_samples(N_test, tau_out, N_modes, nu, x_mesh, t_mesh, seed_test_out, N_cheb);

fprintf('\nTotal generation time: %.1f s\n', toc(tic_total));

fprintf('\nSaving to %s...\n', output_dir);
save_file(fullfile(output_dir, 'viscid_train.mat'),    u0_train,    usol_train,    x_mesh, t_mesh);
save_file(fullfile(output_dir, 'viscid_test_in.mat'),  u0_test_in,  usol_test_in,  x_mesh, t_mesh);
save_file(fullfile(output_dir, 'viscid_test_out.mat'), u0_test_out, usol_test_out, x_mesh, t_mesh);

%% Sanity check
fprintf('\n=== Sanity check ===\n');
fprintf('Training set:\n');
fprintf('  u0    range: [%+.4f, %+.4f],  mean=%.4f,  std=%.4f\n', ...
    min(u0_train(:)), max(u0_train(:)), mean(u0_train(:)), std(u0_train(:)));
fprintf('  u_sol range: [%+.4f, %+.4f],  mean=%.4f\n', ...
    min(usol_train(:)), max(usol_train(:)), mean(usol_train(:)));
fprintf('  u0 at x=-1 (max abs, should be ~0): %.2e\n', max(abs(u0_train(:, 1))));
fprintf('  u0 at x=+1 (max abs, should be ~0): %.2e\n', max(abs(u0_train(:, n_mesh))));
fprintf('  IC consistency max|u_sol(t=0) - u0|: %.2e\n', ...
    max(max(abs(squeeze(usol_train(:, 1, :)) - u0_train))));
fprintf('In-dist test:\n');
fprintf('  u0    mean=%.4f, std=%.4f\n', mean(u0_test_in(:)), std(u0_test_in(:)));
fprintf('Out-of-dist test:\n');
fprintf('  u0    mean=%.4f, std=%.4f\n', mean(u0_test_out(:)), std(u0_test_out(:)));
fprintf('\nDone.\n');


%% =============================================================
%% LOCAL HELPER FUNCTIONS
%% =============================================================

function [u0_all, usol_all] = gen_samples(N, tau, N_modes, nu, x_mesh, t_mesh, seed, N_cheb)
    rng(seed);
    n_mesh = length(x_mesh);
    n_time = length(t_mesh);

    u0_all   = zeros(N, n_mesh,  'double');
    usol_all = zeros(N, n_time, n_mesh, 'double');

    % Precompute GRF coefficient standard deviations
    alpha = 2.5;
    s     = 70.14;
    ns = (1:N_modes)';
    coeff_std = s ./ (tau^2 + (ns * pi).^2).^(alpha/2);

    t_start = tic;
    for n = 1:N
        if mod(n, 50) == 0
            elapsed = toc(t_start);
            fprintf('  [%4d/%4d]  %.1f s elapsed, ~%.1f s remaining\n', ...
                n, N, elapsed, elapsed / n * (N - n));
        end

        % 1. Sample GRF coefficients
        Z = randn(N_modes, 1);
        coeffs = coeff_std .* Z;  % (N_modes, 1)

        % 2. Solve Burgers PDE with Chebfun (IC built analytically)
        u_sol_n = solve_burgers_chebfun(coeffs, x_mesh, t_mesh, nu, N_cheb);  % (n_time, n_mesh)

        % 3. Store: use u_sol(t=0) as u0 so IC is exactly consistent
        u0_all(n, :)      = u_sol_n(1, :);
        usol_all(n, :, :) = u_sol_n;
    end
end


function u_sol = solve_burgers_chebfun(coeffs, x_out, t_mesh, nu, N_cheb)
    % Solve viscous Burgers PDE using Chebfun's pde15s.
    %
    %   u_t = nu * u_xx - u * u_x
    %   BCs: u(-1,t) = u(1,t) = 0  (homogeneous Dirichlet)
    %   IC:  u(x,0) = a(x) = sum_k coeffs(k) * sin(k*pi*x)
    %
    % The IC is constructed analytically as a chebfun from the sine series,
    % avoiding equispaced-to-Chebyshev interpolation errors.
    %
    % Uses Chebyshev spectral collocation with N_cheb points.
    % Returns u_sol (n_time, n_out) evaluated at x_out.
    % u_sol(1,:) is u(x, t=0), which should be used as u0 for exact IC consistency.

    N_modes = length(coeffs);

    % Build initial condition analytically as a chebfun from the sine series.
    % Chebfun adaptively samples the function handle to spectral accuracy.
    u0_cheb = chebfun(@(x) ic_sine_series(x, coeffs, N_modes), [-1, 1]);

    bc.left = 0;
    bc.right = 0;

    % PDE right-hand side: u_t = nu*u_xx - u*u_x
    pdefun = @(t, x, u) nu*diff(u, 2) - u.*diff(u);

    % Solve with fixed spatial resolution, no plotting
    opts = pdeset('Plot', 'off');
    [~, uu] = pde15s(pdefun, t_mesh(:)', u0_cheb, bc, opts, N_cheb);

    % uu is an array-valued chebfun: columns are time slices.
    u_sol = uu(x_out(:));   % (n_out, n_time)
    u_sol = u_sol';          % (n_time, n_out)
end


function save_file(path, u0, u_sol, x_mesh, t_mesh)
    % Save dataset to HDF5-compatible .mat v7.3 file.
    %
    % MATLAB (N, n_mesh)       → h5py reads (n_mesh, N)       → .T → (N, n_mesh)
    % MATLAB (N, n_time, n_mesh) → h5py reads (n_mesh, n_time, N) → .T → (N, n_time, n_mesh)
    % Matches _load_data() in src/problems/burgers.py.

    N  = size(u0, 1);
    mb = (numel(u0) + numel(u_sol)) * 8 / 1e6;
    fprintf('  Saving %s  [N=%d, ~%.0f MB]\n', path, N, mb);
    save(path, 'u0', 'u_sol', 'x_mesh', 't_mesh', '-v7.3');
end


function vals = ic_sine_series(x, coeffs, N_modes)
    % Evaluate a(x) = sum_k coeffs(k) * sin(k*pi*x) at arbitrary points x.
    % Vectorized for Chebfun's adaptive sampling.
    x = x(:);
    ks = (1:N_modes);
    phi = sin(ks .* pi .* x);      % (numel(x), N_modes)
    vals = phi * coeffs(1:N_modes); % (numel(x), 1)
end
