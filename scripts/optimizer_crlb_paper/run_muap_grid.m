%% run_muap_grid.m
% Generates single-MU surface MUAP waveforms with the real Farina & Merletti
% (2001) model at several known conduction velocities, for a 13-channel
% linear array with 8 mm inter-electrode distance and Fs = 2000 Hz -- the
% same array geometry used in the sEMG-demo optimizer/CRLB benchmark
% (src/mfcv.py: GridConfig(fs=2000, ied_m=0.008, n_rows=13)).
%
% This bypasses the full multi-MU recruitment/firing driver (EMG_Model.m /
% Model_Surface_EMG.m) on purpose: for a delay-estimation benchmark we need
% a signal with an UNAMBIGUOUS, EXACTLY KNOWN ground-truth CV, which a
% single MU gives directly (MUAPs(i).CV in the full driver is instead
% randomly assigned per MU across a population).
%
% Usage: edit MODEL_DIR below to point at your local
% "EMG_Model 20180713/EMG_Model" folder, then run this script (Octave or
% MATLAB) with the current directory set to this file's folder
% (scripts/optimizer_crlb_paper/):
%     run_muap_grid
%
% Output: one .mat file per CV value in the current directory, e.g.
% muap_cv3.mat, muap_cv4.mat, muap_cv4p5.mat, muap_cv5.mat, muap_cv6.mat --
% each containing:
%   MUAP    (NChan x w) double  -- monopolar MUAP waveform per channel
%   v, fsamp, dint, NChan, T, h, d, alfa, Nfib, Radius, y0, x0
% Zip these five files and send them back for the Python-side comparison.

clear; clc;

% >>> EDIT THIS to your local path to the "EMG_Model" folder <<<
MODEL_DIR = fullfile('..', '..', 'EMG_Model 20180713', 'EMG_Model');
addpath(MODEL_DIR);

% --- Array geometry: matches the Python benchmark exactly ---
fsamp  = 2000;      % Hz
dint   = 8;          % mm  (IED)
NChan  = 13;         % channels along the fibre direction (n_rows)
NArray = 1;          % single linear array -- we only need propagation
                      % along the fibre direction, not across arrays
darray = 5;           % mm (unused with NArray=1, kept for gen_MUAP's signature)
z0     = 70;           % mm -- offset so the whole 13-channel array (spanning
                        % roughly z0 +/- (NChan-1)/2*dint = 70 +/- 48 mm =
                        % [22, 118] mm) sits on ONE side of the innervation
                        % zone (at z=0) AND well inside the fibre's
                        % propagating region (see L below) -- z0=0 straddles
                        % the IZ, and too-short fibres put the far channels
                        % past the tendon (extinction, not propagation),
                        % both of which badly bias a constant-delay
                        % estimator and are not a fair test.
alfa   = 0;           % fibres parallel to the array
det    = 1;           % monopolar (matches make_column's raw-channel model:
                       % x_k(n) = s(n-k*theta) + w_k(n), no differencing)
electrode = 'circ';
dims   = [3 1];        % electrode radius 3 mm
h = 3; d = 1;           % fat/skin thickness (mm) -- init_EMG_Model.m defaults
T = 50;                 % ms, generous window for slow CV (3 m/s over ~104 mm span)

% --- A single, "typical" MU (init_EMG_Model.m population-average values) ---
Nfib        = 200;
ind_fib_den = 20;
Radius      = sqrt(Nfib/(ind_fib_den*pi));   % mm
y0          = 3;        % mm depth in the muscle
x0          = 0;        % centred under the array
Inn         = 10;        % mm, innervation-zone spread
L           = [150 150]; % mm, semi-fibre lengths -- long enough that the
                          % 96 mm array (z in [22,118]) sits well inside
                          % [Inn/2, L-Ten/2] = [5, 145] mm, away from both
                          % the IZ and the tendon extinction region
Ten         = [10 10];   % mm, tendon-region spread

rng_seed = 42;
if exist('OCTAVE_VERSION', 'builtin') ~= 0
    rand('seed', rng_seed);
else
    rng(rng_seed);
end

cv_values = [3 4 4.5 5 6];
cv_tags   = {'3','4','4p5','5','6'};

for i = 1:numel(cv_values)
    v = cv_values(i);
    fprintf('Generating MU at CV = %.1f m/s ...\n', v);

    [Fibres, Len] = pos_Fibres(Nfib, Radius, y0, x0, Inn, L, Ten);

    MUAP4d = gen_MUAP(Fibres, h, d, alfa, v, dint, darray, fsamp, det, ...
                       Nfib, NChan, NArray, z0, electrode, dims, Len, T);
    MUAP = squeeze(MUAP4d(1,:,:));   % (NChan x w)

    fname = sprintf('muap_cv%s.mat', cv_tags{i});
    save('-v7', fname, 'MUAP', 'v', 'fsamp', 'dint', 'NChan', 'T', 'h', 'd', ...
         'alfa', 'Nfib', 'Radius', 'y0', 'x0');
    fprintf('  wrote %s  (MUAP size: %s)\n', fname, mat2str(size(MUAP)));
end

fprintf('\nDone. Zip muap_cv*.mat and send them back.\n');
