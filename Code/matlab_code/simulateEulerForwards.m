function forwards = simulateEulerForwards(no_sim, no_steps,T,F, alpha, beta, rho, v, r1, r2)
    dt = T / no_steps;
    dt_sqrt = sqrt(dt);
    

    forwards = zeros(no_sim, 1);

    % Generate forwards using Euler discretization. 
    for no_sim_counter = 1:no_sim
        F_t = F;
        alpha_t = alpha;
       
        for no_steps_counter = 1:no_steps
            % forward is zero absorbing. 
            if F_t <= 0
                F_t = 0;
                continue % todo: more efficient when break right?
            end
            
            rand0 = r1(no_sim_counter,no_steps_counter);
            rand1 = r2(no_sim_counter,no_steps_counter);

            dW_F = dt_sqrt * rand0;
            F_t = F_t + alpha_t * (F_t^beta) * dW_F;
            dW_a = dt_sqrt * rand1;
            alpha_t = (alpha_t + v * alpha_t * dW_a);
            
        end

        forwards(no_sim_counter) = F_t;

    end

    
end