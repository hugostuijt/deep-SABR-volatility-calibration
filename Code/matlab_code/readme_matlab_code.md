Matlab code
----------------------------

-   `simulateEulerForwards.m`: simulates SABR forward prices with Euler
    discretization.
-   `simulateEulerIV.m`: This function is used by the `MatlabMonteCarlo`
    class, it is called by the Matlab Python engine. It first simulates
    Euler forwards using the previous function, then computes call
    prices, variances and implied vols using Jaeckel (2015)

