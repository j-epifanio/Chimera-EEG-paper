function [CMF, degreeSync, decayTime] = FocalEEGDrivenChimera_EpsilonDeltaOmega(N, b, alphaD, nEpsilon, nDOmega, ii)
% FocalEEGDrivenChimera_EpsilonDeltaOmega
% ---------------------------------------
% Simulate a 1-layer chimera network driven by a *focal* EEG phase signal
% for a grid of (epsilon, DeltaOmega) values.
%
% For each pair (epsilon, DeltaOmega), the function:
%   - evolves the chimera network driven by the EEG phase
%   - computes:
%       * CMF        : phase-locking value between mean field and driver
%                      (entrainment power in the paper)
%       * degreeSync : average within-network coherence over the
%                      observation window (this is our within-network
%                      coherence measure)
%       * decayTime  : time to full synchronization (R > 0.9999),
%                      referred to as collapse time in the paper.
%
% INPUT:
%   N        : number of oscillators
%   b        : coupling range (number of neighbors on each side)
%   alphaD   : phase lag parameter
%   nEpsilon : number of epsilon values
%   nDOmega  : number of DeltaOmega values
%   ii       : realization index (used to load the corresponding EEG signal)
%
% OUTPUT:
%   CMF        : 1 × (nEpsilon*nDOmega) vector
%   degreeSync : 1 × (nEpsilon*nDOmega) vector, within-network coherence
%   decayTime  : 1 × (nEpsilon*nDOmega) vector, collapse time

%% Time parameters
u  = 3125;
d  = 72;
dt = 0.0025;                     % integration time step

originalSigSamples = 10240;

% Rescale the length of the driving signal
signalSamples  = originalSigSamples*u/d*0.9;
signalDuration = signalSamples*dt;          % duration of the driven part

transientDuration = 300;                   % undriven transient before forcing
startDrive        = transientDuration/dt;  % sample index where driving starts

observationTime  = 1000;                   % length of the evaluation window
observationStart = 500;                    % time from which we start evaluation
T                = signalDuration + transientDuration;  % total simulated time
integrationSamples = T/dt;                 % total number of integration steps

%% Network parameters (one-layer chimera network)
% Phase lag matrix alphaV (all-to-all constant alphaD)
alphaV(1:N,1:N) = alphaD;

% Rectangular coupling kernel: each node coupled to 2b neighbors
Rectangularwindow          = zeros(N, 1);
Rectangularwindow(1:b+1)   = 1;
Rectangularwindow(N-b+1:N) = 1;

% Build coupling matrix G by circularly shifting the window
G = zeros(N, N);
for i = 1:N
    G(i,1:N) = circshift(Rectangularwindow, i-1);
end
G = G/(2*b);   % normalize by number of neighbors

% Differential equation of the network (user-defined external function)
deriv = @AndrzejakChaosNetworkDifferentialEquation;

%% Grid of driving parameters (DeltaOmega and epsilon)
DOmega  = linspace(-1, 1, nDOmega);                 % frequency mismatch
Epsilon = [0 logspace(-2, -0.25, nEpsilon - 1)];    % driving strengths

[DOmega, Epsilon] = meshgrid(DOmega, Epsilon);
Epsilon = reshape(Epsilon, [1, nEpsilon*nDOmega]);  % vectorize the grid
DOmega  = reshape(DOmega,  [1, nEpsilon*nDOmega]);

%% EEG driver: focal channel, resampling, phase, trimming, unwrapping
driver_temp = load( ...
    sprintf('/gpfs/home/jepifanio/Codes/EEG/BernBarcelona/Data_mat_FOCAL/Data_F_Ind%04d.mat', ii) ...
).Data_F;

% Select first channel as driver
driver = driver_temp(1, :);

% Resample from original sampling to target (factor u/d)
driver = resample(driver, u, d);

% Extract instantaneous phase via Hilbert transform
driver = angle(hilbert(driver));

% Cut 5% from both ends to avoid border effects
driver = -driver( ...
    floor(originalSigSamples*u/d*0.05 + 2) : ...
    ceil(originalSigSamples*u/d*0.95) ...
);

% Unwrap phase
driver = unwrap(driver);

%% Driver-based measures (mean instantaneous frequency)
dDri = diff(driver)/dt;      % instantaneous angular frequency
MDri = mean(dDri);           % mean angular velocity of the driver
% (SDri, VDri etc. can be computed if needed)

%% Preallocation
% Time indices where measures are evaluated
I = (observationStart/dt + 1):integrationSamples;

decayTime  = zeros(1, nDOmega*nEpsilon);
CMF        = zeros(1, nDOmega*nEpsilon);
degreeSync = zeros(1, nDOmega*nEpsilon);

%% Main loop over all (epsilon, DeltaOmega) combinations
for i = 1:nDOmega*nEpsilon

    Texternal = tic;

    % Deterministic seed per (epsilon, DeltaOmega, realization) for reproducibility
    rng(nDOmega*nEpsilon*(ii - 1) + i);

    % Phases: rows = oscillators, columns = time
    X = zeros(N, integrationSamples);

    % Parameters for this realization
    epsilon = Epsilon(i);        % driving strength
    omega   = DOmega(i) + MDri;  % natural frequency of the network

    % Flag to avoid initial conditions that lead to immediate synchronization
    fullySynchronized = true;

    %% Undriven transient: find a non-synchronized initial condition
    while fullySynchronized

        % Random initial condition in [-pi, pi)
        X(:, 1) = rand(N, 1)*2*pi - pi;
        x       = X(:, 1);

        % Reset synchronization flag
        fullySynchronized = false;

        % Undriven dynamics (epsilon = 0, input = 0)
        for j = 2:transientDuration/dt

            % Euler step without external drive
            x = x + dt*feval(deriv, x, G, alphaV, zeros(N, 1), 0, 0, omega);

            % Wrap phases to [0, 2*pi)
            x       = mod(x, 2*pi);
            X(:, j) = x;

        end

        % Kuramoto order parameter at the end of the transient
        Zt = mean(exp(1i*X(:, j)), 1);        % complex order parameter
        Rt = (abs(Zt) - 0.1253)/(1 - 0.1253); % normalized R (0 = chimera ref)

        % If the network is already fully synchronized, repeat with new IC
        if Rt > 0.9999
            fullySynchronized = true;
            fprintf('Retrying realization %d due to early synchronization at time %.2f seconds\n', ...
                    i, j * dt);
        end
    end

    % Use last transient state as initial condition for driven dynamics
    x = X(:, transientDuration/dt);

    %% Driven dynamics with EEG driver
    for j = (transientDuration/dt + 1):integrationSamples

        % Euler step with EEG drive
        x = x + dt*feval(deriv, x, G, alphaV, ...
                         driver(j - transientDuration/dt), epsilon, 1, omega);

        % Wrap phases to [0, 2*pi)
        x       = mod(x, 2*pi);
        X(:, j) = x;

    end

    %% Kuramoto order parameters for the full dynamics
    Zt    = mean(exp(1i * X), 1);              % complex order parameter vs time
    Rt    = (abs(Zt) - 0.1253) / (1 - 0.1253); % normalized Kuramoto order parameter
    Phit  = angle(Zt);                         % mean-field phase
    uPhit = unwrap(Phit);                      % unwrapped mean-field phase

    %% Decay time (collapse time in the paper): first time when R > 0.9999
    idxSync = find(Rt > 0.9999, 1);
    if isempty(idxSync)
        % Never reaches full synchrony within simulation
        decayTime(i) = integrationSamples - startDrive;
    else
        decayTime(i) = idxSync - startDrive;
    end

    %% Synchronization measures

    % Phase difference between mean field and driver over the evaluation window
    deltaPhiMF = uPhit(I) - driver(I - startDrive);

    % Entrainment power (PLV between mean field and driver)
    CMF(i) = abs(mean(exp(1i*deltaPhiMF)));

    % Within-network coherence (average normalized Kuramoto order parameter)
    degreeSync(i) = mean(Rt(I));

    fprintf('Realization %d in %f seconds\n', i, toc(Texternal));
end

%% Convert decay time to physical time units
decayTime = decayTime*dt;

end

