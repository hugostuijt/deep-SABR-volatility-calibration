import numpy as np
import scipy.integrate as integrate
import helperfunctions as hf
import time
import joblib
import keras
keras.backend.set_floatx('float64')


def phi(s, s_min, s_plus):
    """
    Antonov (2013) phi function, see page 36 top.
    """
    num = np.sinh(s) ** 2 - np.sinh(s_min) ** 2
    denom = np.sinh(s_plus) ** 2 - np.sinh(s) ** 2
    return 2 * np.arctan(np.sqrt(num / denom))


def psi(s, s_min, s_plus):
    """
    Antonov (2013) psi function, see page 36 top.
    """
    num = np.sinh(s) ** 2 - np.sinh(s_plus) ** 2
    denom = np.sinh(s) ** 2 - np.sinh(s_min) ** 2

    if num == np.inf:
        print('Error in psi')
        num = 1
        denom = 1
    output = 2 * np.arctanh(np.sqrt(num / denom))
    return output


def kernel_approx(tau, s):
    """
    Antonov (2013) approximation of the kernel, see page 36 of the thesis. This is used by the AntonovApprox class
    such that we only have singe integrals.
    """
    coth_s = np.cosh(s) / np.sinh(s)
    g_s = s * coth_s - 1

    frac1 = 3 * tau * g_s / (8 * s ** 2)
    frac2 = 5 * tau ** 2 * (-8 * s ** 2 + 3 * g_s ** 2 + 24 * g_s) / (128 * s ** 4)
    frac3 = 35 * tau ** 3 * (-40 * s ** 2 + 3 * g_s ** 3 + 24 * g_s ** 2 + 120 * g_s) / (1024 * s ** 6)
    R = 1 + frac1 - frac2 + frac3

    frac = (3072 + 384 * tau + 24 * tau ** 2 + tau ** 3) / 3072
    delta_R = np.exp(tau / 8) - frac

    return np.sqrt(np.sinh(s) / s) * np.exp(-s ** 2 / (2 * tau) - tau / 8) * (R + delta_R)


def double_integral1(u, s, eta, tau, s_min, s_plus):
    """
    Antonov (2013) first double integral for call option price, see Equation (13) on page 35. This is without the
    kernel approximation, thus used by the AntonovExact class. The kernel is thus g_us.
    """
    a_s = np.sin(eta * phi(s, s_min, s_plus)) / np.sinh(s)
    b_tau = 2 * np.sqrt(2) * np.exp(-tau / 8) / (tau * np.sqrt(2 * np.pi * tau))
    g_us = u * np.exp(-u * u / (2 * tau)) * np.sqrt(np.cosh(u) - np.cosh(s))

    return a_s * b_tau * g_us


def double_integral2(u, s, eta, tau, s_min, s_plus):
    """
    Antonov (2013) second double integral for call option price, see Equation (13) on page 35. This is without the
    kernel approximation, thus used by the AntonovExact class. The kernel is thus g_us.
    """
    a_s = np.exp(-eta * psi(s, s_min, s_plus)) / np.sinh(s)
    b_tau = 2 * np.sqrt(2) * np.exp(-tau / 8) / (tau * np.sqrt(2 * np.pi * tau))
    g_us = u * np.exp(-u * u / (2 * tau)) * np.sqrt(np.cosh(u) - np.cosh(s))

    return a_s * b_tau * g_us


def integral1_approx(s, eta, tau, s_min, s_plus):
    """
    Antonov (2013) first integral in call option price. This is with the kernel approximation, thus used in the
    AntonovApprox class.
    """
    frac = np.sin(eta * phi(s, s_min, s_plus)) / np.sinh(s)

    return frac * kernel_approx(tau, s)


def integral2_approx(s, eta, tau, s_min, s_plus):
    """
    Antonov (2013) second integral in call option price. This is with the kernel approximation, thus used in the
    AntonovApprox class.
    """
    frac = np.exp(-eta * psi(s, s_min, s_plus)) / np.sinh(s)

    return frac * kernel_approx(tau, s)


def calc_efficient_params(v0, beta, rho, gamma, contract_params, T):
    """
    Calculation of the Antonov (2013) parameters for the mimicking model with zero correlation. We use the Antonov notation
    in the code (less prone to making errors).
    :param v0: SABR alpha
    :param beta: SABR beta
    :param rho: SABR rho
    :param gamma: SABR vol of vol v
    """
    K, F = hf.get_contract_parameters(contract_params)

    beta_tilde = beta
    gamma_tilde_squared = gamma ** 2 - 3 * ((gamma * rho) ** 2 + v0 * gamma * rho * (1 - beta) * (F ** (beta - 1))) / 2
    gamma_tilde = np.sqrt(gamma_tilde_squared)

    delta_q = ((K ** (1 - beta)) - (F ** (1 - beta))) / (1 - beta)
    v_min_squared = (gamma * delta_q) ** 2 + 2 * rho * gamma * delta_q * v0 + v0 ** 2
    v_min = np.sqrt(v_min_squared)

    # calculation of Phi
    num = v_min + rho * v0 + gamma * delta_q
    denom = v0 * (1 + rho)
    Phi = (num / denom) ** (gamma_tilde / gamma)

    delta_q_tilde = ((K ** (1 - beta_tilde)) - (F ** (1 - beta_tilde))) / (1 - beta_tilde)

    # calculation of v0_tilde
    num = 2 * Phi * delta_q_tilde * gamma_tilde
    denom = Phi ** 2 - 1
    v0_tilde_0 = num / denom

    # calculation of u0
    num = delta_q * gamma * rho + v0 - v_min
    denom = delta_q * gamma * np.sqrt(1 - rho * rho)
    u0 = num / denom

    q = K ** (1 - beta) / (1 - beta)  # edit
    # calculation of L
    num = v_min  # * (1 - beta)
    denom = (K ** (1 - beta)) * gamma * np.sqrt(1 - rho * rho)
    denom = q * gamma * np.sqrt(1 - rho * rho)
    L = num / denom

    # calculation of I
    if L < 1:
        denom = np.sqrt(1 - L * L)
        I_f = (2 / denom) * (np.arctan((u0 + L) / denom) - np.arctan(L / denom))
    elif L > 1:
        denom = np.sqrt(L * L - 1)
        I_f = (1 / denom) * np.log((u0 * (L + denom) + 1) / (u0 * (L - denom) + 1))
    else:
        print(v0)
        print(beta)
        print(rho)
        print(gamma)
        print(contract_params)
        print(T)
        raise Exception('Errors with calculation of L')

    # calculation of phi_0
    num = delta_q * gamma + v0 * rho
    phi0 = np.arccos(-num / v_min)

    B_min = -0.5 * (beta / (1 - beta)) * (rho / np.sqrt(1 - rho * rho)) * (np.pi - phi0 - np.arccos(rho) - I_f)

    # formulas from Antonov (2013) which do not work.

    # calculation of v0_tilde_1
    # denom = np.log(Phi) * ((Phi*Phi - 1) / (Phi * Phi + 1))
    # num1 = 0.5 * (beta - beta_tilde) * np.log(K*F) + 0.5 * np.log(v0 * v_min)
    # frac1 = num1 / denom
    # num2 = 0.5 * np.log(v0_tilde_0 * np.sqrt((delta_q_tilde * gamma_tilde)**2 + v0_tilde_0*v0_tilde_0)) - B_min
    # frac2 = num2 / denom

    # v0_tilde_1 = v0_tilde_0 * gamma_tilde * gamma_tilde * (frac1 - frac2)

    # Formulas from Antonov (2015) which do work, see footnote 5 on page (35).
    v_min_tilde2 = (gamma_tilde * delta_q) ** 2 + v0_tilde_0 ** 2
    v_min_tilde = np.sqrt(v_min_tilde2)
    R_tilde = delta_q * gamma_tilde / v0_tilde_0
    h = np.sqrt(1 + R_tilde ** 2)
    num = 0.5 * np.log(v0 * v_min / (v0_tilde_0 * v_min_tilde)) - B_min
    denom = R_tilde * np.log(h + R_tilde)
    v0_tilde_1 = v0_tilde_0 * (gamma_tilde ** 2) * h * num / denom

    v0_tilde = v0_tilde_0 + T * v0_tilde_1

    return v0_tilde, beta_tilde, gamma_tilde


def elu(x):
    """
    Elu activation function, obtained using Horvath (2021) code.
    """
    ind = (x < 0)
    x[ind] = np.exp(x[ind]) - 1
    return x


class ImpliedVolatilityApproximator:
    """
    Class for SABR implied vol approximators. We have a function to calculate the IV, the call price, and to retrieve
    the name of the approximator. We have children classes for the Hagan (2002) formulas (both normal and lognormal IV),
    Obloj correction (see beginning of the literature), Antonov (2013) approximations (both the exact version, and the
    version with the kernel approximation used in the research), and the Antonov Neural network approximation.
    """

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        pass

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        pass

    def get_name(self):
        pass


class HaganNormal(ImpliedVolatilityApproximator):
    """
    Hagan (2002) normal implied volatility approximation. Only used for testing/debugging purposes.
    """

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        l_fk = np.log(F / K)

        z = (v / alpha) * l_fk * (K * F) ** ((1 - beta) / 2)

        num = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
        zeta = np.log(num / (1 - rho))

        num1 = 1 + l_fk ** 2 / 24 + l_fk ** 4 / 1920
        denom1 = 1 + l_fk ** 2 * (1 - beta) ** 2 / 24 + l_fk ** 4 * (1 - beta) ** 4 / 1920
        frac1 = num1 / denom1

        part1 = (-beta * (2 - beta) / 24) * (alpha ** 2 / ((K * F) ** (1 - beta))) * T
        part2 = .25 * (alpha * beta * rho * v * T) / ((K * F) ** ((1 - beta) / 2))
        part3 = (2 - 3 * rho ** 2) * v ** 2 * T / 24

        iv = alpha * (K * F) ** (beta / 2) * frac1 * (z / zeta) * (1 + part1 + part2 + part3)

        black_iv = iv * (l_fk / (F - K)) * (1 + iv ** 2 * T / (24 * F))

        return black_iv

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        return np.nan

    def get_name(self):
        return 'Hagan Normal Vol'


class Hagan(ImpliedVolatilityApproximator):
    """
    Hagan (2002) implied volatility approximation.
    """

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        l_fk = np.log(F / K)
        z = l_fk * (v / alpha) * (F * K) ** ((1 - beta) / 2)
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        q = (1 + (1 - beta) ** 2 * l_fk ** 2 / 24 + (1 - beta) ** 4 * l_fk ** 4 / 1920) * (F * K) ** ((1 - beta) / 2)

        if F == K:
            iv = (alpha / F ** (1 - beta)) * (1 + T * (((1 - beta) ** 2 / 24) * (
                    alpha ** 2 / F ** (2 - 2 * beta)) + .25 * rho * beta * v * alpha / F ** (1 - beta) + (
                                                               2 - 3 * rho * rho) * v * v / 24))
            return iv

        iv = (alpha / q) * (z / x_z) * (1 + T * (
                ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F * K) ** (1 - beta)) + .25 * rho * beta * v * alpha / (
                F * K) ** ((1 - beta) / 2) + (2 - 3 * rho * rho) * v * v / 24))

        return iv

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        return np.nan

    def get_name(self):
        return 'Hagan'


class Obloj(ImpliedVolatilityApproximator):
    """
    Obloj (2008) implied volatility approximation (see first paragraph of the literature section). Only used for
    testing/debugging purposes.
    """

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        if F == K:
            iv = (alpha / F ** (1 - beta)) * (1 + T * (((1 - beta) ** 2 / 24) * (
                    alpha ** 2 / F ** (2 - 2 * beta)) + .25 * rho * beta * v * alpha / F ** (1 - beta) + (
                                                               2 - 3 * rho * rho) * v * v / 24))
            return iv

        l_fk = np.log(F / K)
        zeta = (v / alpha) * (F ** (1 - beta) - K ** (1 - beta)) / (1 - beta)
        num = np.sqrt(1 - 2 * rho * zeta + zeta * zeta) + zeta - rho
        I0 = v * l_fk / np.log(num / (1 - rho))

        frac1 = (1 - beta) ** 2 / 24
        frac2 = alpha ** 2 / ((F * K) ** (1 - beta))
        frac3 = (rho * beta * alpha * v) / ((F * K) ** (0.5 * (1 - beta)))
        frac4 = (2 - 3 * rho ** 2) * v ** 2 / 24

        I1 = frac1 * frac2 + .25 * frac3 + frac4

        iv = I0 * (1 + I1 * T)

        return iv

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        return np.nan

    def get_name(self):
        return 'Obloj'


class AntonovApprox(ImpliedVolatilityApproximator):
    """
    Antonov (2013) implied volatility approximation with approximated kernel function. This class therefore consists
    of single integrals such that it is much faster than the AntonovExact class (which has double integrals).
    """

    def get_name(self):
        return 'Antonov Approx'

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        # Switch to Antonov (2013) notation
        gamma = v
        v0 = alpha

        # Calculate the parameters of the mimicking zero correlation SABR model.
        if rho != 0:
            v0, beta, gamma = calc_efficient_params(v0, beta, rho, gamma, contract_params, T)

        # call option approximation
        q = (K ** (1 - beta)) / (1 - beta)
        q0 = (F ** (1 - beta)) / (1 - beta)

        s_min = np.arcsinh((gamma * np.abs(q - q0)) / v0)
        s_plus = np.arcsinh((gamma * (q + q0)) / v0)

        eta = np.abs(1 / (2 * (beta - 1)))
        tau = T * gamma * gamma

        if np.nan in [s_min, s_plus, eta, v]:
            return np.inf

        # compute the integrals
        integral1_eval = integrate.quad(lambda s: integral1_approx(s, eta, tau, s_min, s_plus), a=s_min, b=s_plus)
        integral2_eval = integrate.quad(lambda s: integral2_approx(s, eta, tau, s_min, s_plus), a=s_plus, b=200)

        call_price = max(F - K, 0) + (2 / np.pi) * np.sqrt(K * F) * (
                integral1_eval[0] + np.sin(eta * np.pi) * integral2_eval[0])

        return call_price

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        # if K \approx f, the approximation is undefined. We take two points around the considered strike and use
        # linear interpolation
        if np.round(K, 2) == float(F):
            offset = .006
            K2 = (1 + offset) * F
            K1 = (1 - offset) * F
            iv1 = self.calc_iv(alpha, beta, rho, v, (K1, F), T)
            iv2 = self.calc_iv(alpha, beta, rho, v, (K2, F), T)

            helling = (iv2 - iv1) / (K2 - K1)
            return helling * (K - K1) + iv1

        # calculate call price and its IV using Jaeckel (2015).
        call_price = self.calc_call(alpha, beta, rho, v, contract_params, T)
        iv = hf.calc_black_scholes_iv(call_price, (K, F), T)
        return iv


class AntonovANN(ImpliedVolatilityApproximator):
    """
    Neural network approximation of the Antonov (2013) function.
    """

    def scale_y_inverse_transform(self, y):
        return self.scale_y.inverse_transform(y)

    def NeuralNetwork(self, x):
        """
        Compute the neural network estimate of the input parameters using numpy. This is much faster than the keras
        prediction functions. This is obtained using Horvath (2021) code.
        :param x: network input
        :return: network output
        """
        input1 = x
        for i in range(self.NumLayers):
            input1 = np.dot(input1, self.NNParameters[i][0]) + self.NNParameters[i][1]
            # Elu activation
            input1 = elu(input1)

        i += 1
        return np.dot(input1, self.NNParameters[i][0]) + self.NNParameters[i][1]

    def __init__(self):
        """
        Create the full network in the constructor
        """
        # load the standard scaler obtained in network training
        path = r'C:\Users\hugo\OneDrive\Documents\Quantitative Finance\Thesis\Data\ann'
        self.scale_x = joblib.load(path + '\scale_x_pointwise.bin')
        self.scale_y = joblib.load(path + '\scale_y_pointwise.bin')

        input_params = keras.layers.Input(shape=(5,))  # inpute layer

        # four hidden layers with elu activation
        hidden_1 = keras.layers.Dense(30, activation='elu')(input_params)
        hidden_2 = keras.layers.Dense(30, activation='elu')(hidden_1)
        hidden_3 = keras.layers.Dense(30, activation='elu')(hidden_2)
        hidden_4 = keras.layers.Dense(30, activation='elu')(hidden_3)

        # output layer with linear activation
        output_layer = keras.layers.Dense(1, activation='linear')(hidden_4)

        # create keras model
        self.model = keras.models.Model(inputs=input_params, outputs=output_layer)
        self.model.load_weights(path + '\weights_pointwise.h5')
        self.NNParameters = []
        for i in range(1, len(self.model.layers)):
            self.NNParameters.append(self.model.layers[i].get_weights())
        self.NumLayers = 4

    def get_name(self):
        return 'Antonov ANN'

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        """
        This has not been created, although it is possible to retrieve the call price via Black when we
        compute the IV first.
        """
        return np.nan

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        params = np.array([alpha, rho, v, T, K]).reshape(1, -1)

        iv = self.scale_y_inverse_transform(self.NeuralNetwork(self.scale_x.transform(params)))
        return float(iv)


class AntonovExact(ImpliedVolatilityApproximator):
    """
    Antonov (2013) implied volatility approximation without kernel function. This class therefore consists
    of double integrals. It is only used for testing/debugging purposes (it has basically the same accuracy as
    AntonovApprox, but it's much slower).
    """

    def get_name(self):
        return 'Antonov Exact'

    def calc_call(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        # at the money case
        if K == F:
            offset = .01
            K1 = (1 + offset) * K
            K2 = (1 - offset) * K
            call1 = self.calc_call(alpha, beta, rho, v, (K1, F), T)
            call2 = self.calc_call(alpha, beta, rho, v, (K2, F), T)
            return np.mean([call1, call2])

        # Switch to Antonov (2013) notation
        gamma = v
        v0 = alpha

        if rho != 0:
            v0, beta, gamma = calc_efficient_params(v0, beta, rho, gamma, contract_params, T)

        # call option approximation
        q = (K ** (1 - beta)) / (1 - beta)
        q0 = (F ** (1 - beta)) / (1 - beta)

        s_min = np.arcsinh((gamma * np.abs(q - q0)) / v0)
        s_plus = np.arcsinh((gamma * (q + q0)) / v0)

        eta = np.abs(1 / (2 * (beta - 1)))
        tau = T * gamma * gamma

        if np.nan in [s_min, s_plus, eta, v]:
            return np.inf

        # compute the double integrals in the call option price.
        start = time.time()
        integral1_eval = integrate.romberg(
            lambda s: integrate.quad(lambda u: double_integral1(u, s, eta, tau, s_min, s_plus), s, s + 15)[0], s_min,
            s_plus)
        integral2_eval = integrate.romberg(
            lambda s: integrate.quad(lambda u: double_integral2(u, s, eta, tau, s_min, s_plus), s, s + 15)[0], s_plus,
            15)
        print('Integrals: ' + str(time.time() - start))

        call_price = max(F - K, 0) + (2 / np.pi) * np.sqrt(K * F) * (
                integral1_eval + np.sin(eta * np.pi) * integral2_eval)

        return call_price

    def calc_iv(self, alpha, beta, rho, v, contract_params, T):
        K, F = hf.get_contract_parameters(contract_params)

        call_price = self.calc_call(alpha, beta, rho, v, contract_params, T)

        iv = hf.calc_black_scholes_iv(call_price, (K, F), T)

        return iv
