function [iv, forward, sigma] = simulateEulerIV(no_sim, no_steps,T ,r, F, alpha, beta, rho, v, strike_min, strike_max, strike_step, seed)
    % this function is called using the Matlab Python engine. It first
    % simulates no_sim forwards, and next computes the implied vol using
    % Jaeckel (2015). 
    
    rng(seed)    

    % Create correlated normal random variables
    r1 = normrnd(0, 1, no_sim, no_steps);
    y1 = normrnd(0, 1, no_sim, no_steps);
    r2 = r1 * rho + y1 * sqrt(1-rho^2);
    
    forwards = simulateEulerForwards(no_sim, no_steps,T,F, alpha, beta, rho, v, r1, r2);
    
    % vector with strike prices corresponding to bounds/step size
    strikes = strike_min:strike_step:strike_max;
    
    % compute call option prices for all simulated forwards
    payoff = ones(no_sim, length(strikes)) .* strikes;
    payoff = forwards - payoff;
    payoff = max(payoff, 0);
    call_price = exp(-r * T) * payoff;
    
    sigma = var(call_price);    % variance used for the Bayesian calibration
    call_price = mean(payoff);
    forward = mean(forwards);
    iv = blsimpv(forward, strikes, 0, T, call_price, 'Method', 'jackel2016');
    
end
