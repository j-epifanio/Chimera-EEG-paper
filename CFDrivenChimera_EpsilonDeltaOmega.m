function [CMF, S, degreeSync, decayTime] = CFDrivenChimera_EpsilonDeltaOmega(N, b, alphaD, nEpsilon, nDOmega, ii)

% Simulate a 1-layer chimera network driven by a constant-frequency phase
% for a grid of (epsilon, DeltaOmega) values. To get the figure in the
% paper, you will need a number of different grids equal to the number
% of realizations considered (in this case, 100; that is, ii = 1:100).
%
% Input:
%   N        : number of oscillators
%   b        : coupling range (number of neighbors on each side)
%   alphaD   : phase lag parameter
%   nEpsilon : number of epsilon values
%   nDOmega  : number of DeltaOmega values
%   ii       : realization index 
%
% Outputs:
%   CMF        : phase-locking value between the network mean phase and the driver
%                for each (epsilon, DeltaOmega) (entrainment power in the paper).
%   S          : normalized accumulated phase-difference index
%                for each (epsilon, DeltaOmega).
%   degreeSync : average within-network coherence over the observation window
%   decayTime  : time needed for the network to reach full synchronization
%                (R > 0.9999) for each (epsilon, DeltaOmega). It is useful
%                to compute the collapse power, i.e., the fraction of
%                realizations that collapse within 1000 units of time.

%% Time parameters
u  = 3125;
d  = 72;
dt = 0.0025;                   % integration time step

originalSigSamples = 10240;
% Rescale the length of the driving signal
signalSamples  = originalSigSamples*u/d*0.9;
signalDuration = signalSamples*dt;    % duration of the driven part

transientDuration = 300;              % undriven transient before forcing
startDrive        = transientDuration/dt;  % sample index where driving starts

observationTime  = 1000;              % duration used to evaluate measures
observationStart = 500;               % time from which we evaluate measures
T                = signalDuration + transientDuration;  % total simulated time
integrationSamples = T/dt;            % total number of integration steps

%% Network parameters (one-layer chimera network)
% Phase lag matrix alphaV (all-to-all constant alphaD)
alphaV(1:N,1:N) = alphaD;

% Rectangular coupling kernel: each node coupled to 2b neighbors
Rectangularwindow           = zeros(N, 1);
Rectangularwindow(1:b+1)    = 1;
Rectangularwindow(N-b+1:N)  = 1;

% Build coupling matrix G by circularly shifting the window
G = zeros(N, N);
for i = 1:N
    G(i,1:N) = circshift(Rectangularwindow, i-1);
end
G = G/(2*b);   % normalize by number of neighbors

% Differential equation of the network (user-defined external function)
deriv = @EpifanioPRENetworkDifferentialEquation;

%% Grid of driving parameters (DeltaOmega and epsilon)
DOmega  = linspace(-1, 1, nDOmega);                 % frequency mismatch
Epsilon = [0 logspace(-2, -0.25, nEpsilon - 1)];    % driving strengths

[DOmega, Epsilon] = meshgrid(DOmega, Epsilon);
Epsilon = reshape(Epsilon, [1, nEpsilon*nDOmega]);  % vectorize the grid
DOmega  = reshape(DOmega,  [1, nEpsilon*nDOmega]);

%% Preallocation of measures
% Time indices where measures are evaluated
I = (observationStart/dt + 1):integrationSamples;

CMF        = zeros(1, nDOmega*nEpsilon);   % entrainment power
S          = zeros(1, nDOmega*nEpsilon);   % normalized accumulated phase-difference index
degreeSync = zeros(1, nDOmega*nEpsilon);   % within-network coherence
decayTime  = zeros(1, nDOmega*nEpsilon);   % collapse time

%% Main loop over all (epsilon, DeltaOmega) combinations
for i = 1:nDOmega*nEpsilon

    Texternal = tic;

    % Deterministic seed per (epsilon, DeltaOmega, realization) for reproducibility
    rng(nDOmega*nEpsilon*(ii - 1) + i);

    % Phases: rows = oscillators, columns = time
    X = zeros(N, integrationSamples);

    %% External driving signal: constant-frequency phase
    Omega  = -0.5;  % angular frequency of the driver 
    driver = Omega*(dt:dt:observationTime);
    
    % Parameters for this realization
    epsilon = Epsilon(i);        % driving strength
    omega   = DOmega(i) + Omega; % natural frequency of the network

    % Flag to avoid using initial conditions that lead to immediate
    % full synchronization during the undriven transient
    fullySynchronized = true;

    %% Find a non-synchronized initial state (undriven transient)
    while fullySynchronized

        % Random initial condition in [-pi, pi)
        X(:, 1) = rand(N, 1)*2*pi - pi;
        x       = X(:, 1);

        % Reset synchronization flag
        fullySynchronized = false;

        % Undriven dynamics during the transient
        for j = 2:transientDuration/dt

            % Euler step without external drive (epsilon = 0, input = 0)
            x = x + dt*feval(deriv, x, G, alphaV, zeros(N, 1), 0, 0, omega);

            % Wrap phases to [-pi, pi)
            x       = wrapToPi(x);
            X(:, j) = x;

        end

        % Kuramoto order parameter at the end of the transient
        Zt = mean(exp(1i*X(:, j)), 1);        % complex order parameter
        Rt = (abs(Zt) - 0.1253)/(1 - 0.1253); % normalized R (0 = chimera ref)

        % If the network is already fully synchronized, repeat with new
        % initial condition
        if Rt > 0.9999
            fullySynchronized = true;
            fprintf('Retrying realization %d due to early synchronization at time %.2f seconds\n', ...
                    i, j * dt);
        end
    end

    % Use last transient state as initial condition for driven dynamics
    x = X(:, transientDuration/dt);

    %% Driven dynamics
    for j = (transientDuration/dt + 1):integrationSamples

        % Euler step with external drive and coupling strength epsilon
        % driver index shifted by transientDuration/dt
        x = x + dt*feval(deriv, x, G, alphaV, ...
                         driver(j - transientDuration/dt), epsilon, 1, omega);

        % Wrap phases to [-pi, pi)
        x       = wrapToPi(x);
        X(:, j) = x;

    end

    %% Kuramoto order parameters for the full trajectory
    Zt   = mean(exp(1i * X), 1);              % complex order parameter vs time
    Rt   = (abs(Zt) - 0.1253) / (1 - 0.1253); % normalized Kuramoto order parameter
    Phit = angle(Zt);                          % mean-field phase
    uPhit = unwrap(Phit);                     % unwrapped mean-field phase

    %% Decay time to full synchronization (R > 0.9999)
    idxSync = find(Rt > 0.9999, 1);     % first time of full synchrony
    if isempty(idxSync)
        % Never reaches full synchrony within simulation
        decayTime(i) = integrationSamples - startDrive;
    else
        decayTime(i) = idxSync - startDrive;
    end

    %% Measures relative to the external driver
    % Phase difference between mean field and driver over evaluation window
    deltaPhiMF = uPhit(I) - driver(I - startDrive);

    % Entrainment power
    CMF(i) = abs(mean(exp(1i*deltaPhiMF)));

    % Normalized accumulated phase-difference index
    S(i) = (max(deltaPhiMF) - min(deltaPhiMF)) / (2*pi);

    % Average within-network coherence over time
    degreeSync(i) = mean(Rt(I));

    % Convert decay time to physical time units
    decayTime(i) = decayTime(i)*dt;

    fprintf('Realization %d. in %f\n', i, toc(Texternal));
end
