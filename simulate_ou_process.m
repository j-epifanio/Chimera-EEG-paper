function X = simulate_ou_process(theta, mu, sigma, X0, T, dt)
    % simulate_ou_process
    % -------------------
    % Simulates a 1D Ornstein–Uhlenbeck process using the Euler–Maruyama scheme:
    %
    %   dX_t = theta * (mu - X_t) dt + sigma dW_t
    %
    % INPUT:
    %   theta : mean-reversion rate (> 0)
    %   mu    : long-term mean of the process
    %   sigma : noise intensity (standard deviation of the stochastic term)
    %   X0    : initial value X(0)
    %   T     : total simulation time
    %   dt    : integration time step
    %
    % OUTPUT:
    %   X     : 1×n_samples vector containing the simulated trajectory X(t)

    % Number of samples (so that n_samples*dt ≈ T)
    n_samples = round(T / dt);
    
    % Preallocate output
    X = zeros(1, n_samples);
    X(1) = X0;  % Initial condition

    % Generate Wiener increments ~ N(0, dt)
    dW = sqrt(dt) * randn(1, n_samples - 1);

    % Euler–Maruyama integration for the OU process
    for i = 2:n_samples
        X(i) = X(i - 1) ...
             + theta * (mu - X(i - 1)) * dt ...   % drift towards the mean
             + sigma * dW(i - 1);                 % stochastic term
    end
end
