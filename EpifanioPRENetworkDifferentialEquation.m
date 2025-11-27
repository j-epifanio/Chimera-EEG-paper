function dy = EpifanioPRENetworkDifferentialEquation(x, G, alpha, wn, epsilon, sigma, omega)
    % EpifanioPRENetworkDifferentialEquation
    % --------------------------------------
    % Right-hand side of the phase dynamics for a chimera-like network
    % driven by an external signal.
    %
    % INPUTS
    %   x       : N×1 vector of oscillator phases φ_j(t)
    %   G       : N×N coupling matrix (topology + coupling weights)
    %   alpha   : phase-lag parameter (scalar)
    %   wn      : N×1 vector with the driving signal (e.g. η_j(t))
    %   epsilon : driving strength ε
    %   sigma   : scaling factor for the driving signal
    %   omega   : intrinsic angular frequency ω (scalar)
    %
    % OUTPUT
    %   dy      : N×1 vector with the time derivative dφ_j/dt

    % Pairwise phase differences: diffm(j,k) = x(j) - x(k)
    diffm = bsxfun(@minus, x, x');

    % Network term:
    %   -sum_k G(j,k) * sin( x(j) - x(k) + alpha )
    % External driving term:
    %   -epsilon * sin( x(j) - sigma * wn(j) )
    %
    dy = omega - sum( sin(diffm + alpha) .* G, 2 )- epsilon * sin( x - sigma * wn );
end
