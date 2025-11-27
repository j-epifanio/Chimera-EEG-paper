clear all;
close all;
clc;

%% ---------------- Parameters ----------------

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

%% Network parameters (you must set these!)
N      = 50;       % number of oscillators (example)
b      = 18;       % coupling range (neighbors on each side, example)
alphaD = 1.46;     % phase lag parameter (example)

%% Driving parameters
epsilon = 0.01;
DOmega  = 0.33;

%% ---------------- Network construction ----------------

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
deriv = @EpifanioPRENetworkDifferentialEquation;

%% ---------------- Simulation ----------------

% (We could define I for later measures; here it is not strictly needed)
% I = (observationStart/dt + 1):integrationSamples;

Texternal = tic;

% Phases: rows = oscillators, columns = time
X = zeros(N, integrationSamples);

%% External driving signal: constant-frequency phase
Omega  = -0.5;                          % angular frequency of the driver 
driver = Omega*(dt:dt:observationTime); % driver(t) = Omega * t

% Natural frequency of the network
omega = - DOmega + Omega +0.83;

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

    % If the network is already fully synchronized, repeat with new IC
    if Rt > 0.9999
        fullySynchronized = true;
        fprintf('Retrying due to early synchronization at time %.2f seconds\n', j * dt);
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
Zt    = mean(exp(1i * X), 1);              % complex order parameter vs time
Rt    = (abs(Zt) - 0.1253) / (1 - 0.1253); % normalized Kuramoto order parameter
Phit  = angle(Zt);                         % mean-field phase
uPhit = unwrap(Phit);                      % unwrapped mean-field phase

% Mean-field angular velocity
duPhit = diff(uPhit)/dt;

% Node-wise instantaneous angular velocities
V = diff(unwrap(X, [], 2), 1, 2)/dt;      % size N × (integrationSamples - 1)

%% ---------------- Plotting ----------------

figure('Units', 'centimeters', 'Position', [1, 1, 7, 18]);
set(groot, 'DefaultTextInterpreter', 'latex');  
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');  
set(groot, 'DefaultLegendInterpreter', 'latex');  

tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Time vectors for plotting
t_full = dt:dt:T;             % length = integrationSamples
t_vel  = dt:dt:(T - dt);      % length = integrationSamples - 1
t_du   = 2*dt:dt:T;           % length = integrationSamples - 1

%% (a) Phase velocities V_j(t)
nexttile;
imagesc(t_vel, 1:N, V);
title('(a)');
ax1 = gca;
ax1.TitleHorizontalAlignment = 'left';
ylabel('$j$');
xline(300, 'Color', 'green', 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xline(500, 'Color', [0.06, 1, 1], 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xticks(ax1, [1 650 1299]);
xticklabels(ax1, {'', '', ''});   % no labels (only ticks)

yticks(ax1, [10 30 50]);
yticklabels(ax1, {'10', '30', '50'});

%% (b) Normalized Kuramoto order parameter R(t)
nexttile;
plot(t_full, Rt, 'Color', 'k');
ax3 = gca;
title('(b)');
ax3.TitleHorizontalAlignment = 'left';
ylabel('$\rho(t)$');
ylim([-0.05 1.05]);
xlim([0 1300]);
xline(300, 'Color', 'green', 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xline(500, 'Color', [0.06, 1, 1], 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xticks(ax3, [1 650 1299]);
xticklabels(ax3, {'', '', ''});
yticks(ax3, [0 1]);
yticklabels(ax3, {'0', '1'});

%% (c) Mean-field phase Phi(t)
nexttile;
plot(t_full, uPhit);
ax5 = gca;
set(gca, 'YTick', []);
title('(c)');
ax5.TitleHorizontalAlignment = 'left'; 
ylabel('$\Phi(t)$');
xlim([0 1300]);
%ylim([-1100 1.1]);
xline(300, 'Color', 'green', 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xline(500, 'Color', [0.06, 1, 1], 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xticks(ax5, [1 650 1299]);
xticklabels(ax5, {'', '', ''});
yticks(ax5, [-1000 0]);
yticklabels(ax5, {'-1000', '0'});

%% (d) Mean-field angular velocity dPhi/dt
nexttile;
plot(t_du, duPhit);
ax7 = gca;
set(gca, 'YTick', []);
xlabel('$t$ [a.u.]');
title('(d)');
ax7.TitleHorizontalAlignment = 'left'; 
ylabel('$\dot{\Phi}(t)$');
xlim([0 1300]);
xline(300, 'Color', 'green', 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xline(500, 'Color', [0.06, 1, 1], 'LineWidth', 1.5, 'Alpha', 1, 'LineStyle', '--');
xticks(ax7, [1 650 1299]);
xticklabels(ax7, {'0', '650', '1300'});
%ylim([-1.05 -0.4]);
yticks(ax7, [-1.02 -0.83 -0.5]);
yticklabels(ax7, {'-1.02', '-0.83', '-0.5'});

fprintf('Simulation completed in %f seconds.\n', toc(Texternal));
