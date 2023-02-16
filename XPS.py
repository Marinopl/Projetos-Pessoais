import numpy as np
from numpy import linalg as la
import gmpy2
import matplotlib.pyplot as plt
import scipy.optimize as opt

T = 1

def lamb() -> float:
    """Common ratio parameter: lambda = 2"""
    return 2.0


def int_w():
    return np.power(lamb(), np.linspace(-1, 1, 1000))


def int_epsilon():
    return np.power(10, np.linspace(-2, -5, 100))


def zeta():
    """" returns offset """
    return 0


def t_n(n, w):
    """Hopping energy of eNRG parametrization: n = number of states"""

    return T / w * np.power(lamb(), -n - 1 / 2)


def normaliz(n, w):
    """returns RG renormalization factor, equal to smallest codiagonal element"""
    return t_n(n - 2, w)


def dig_ham(n, k, w):
    """Hamiltonian: n(odd) = Matrix Dimension / k = scattering potential"""
    if n % 2 == 0:
        print("Sorry: n must be odd")
        exit(1)

    if zeta() == 0:
        ham__ = np.zeros((n + 1, n + 1))
        ham__[0][0] = k
        ham__[0][1] = ham__[1][0] = T / np.sqrt(w)
        for i in range(1, (n + 1) - 1):
            ham__[i][i + 1] = ham__[i + 1][i] = t_n(i - 1, w)
        eval_, evec__ = la.eigh(ham__)
        return eval_, evec__.T, eval_ / normaliz(n, w)

    else:
        """returns Hamiltonian with offset"""
        ham__ = np.zeros((n + 1, n + 1))
        ham__[zeta()][zeta()] = k
        ham__[0][zeta()] = ham__[zeta()][0] = T
        for i in range(zeta()):
            ham__[0][i] = ham__[i][0] = 0
        for i in range(zeta(), (n + 1) - 1):
            ham__[i][i + 1] = ham__[i + 1][i] = t_n(i - (1+zeta()), w)
        eval_, evec__ = la.eigh(ham__)
        return eval_, evec__.T, eval_ / normaliz(n, w)


def delta(n, k, w):
    """"returns phase shift / pi associated with potential scattering k"""
    ene0_ = dig_ham(n, 0, w)[2]
    ene_ = dig_ham(n, k, w)[2]
    nfermi3: int = int((n + 1) / 2 + 3)
    ret = np.log(ene_[nfermi3] / ene0_[nfermi3]) / np.log(np.power(lamb(), 2.0))

    return ret


def xps_proj(n, k, w, hole, head):
    """Determinant of a projection of the final state over the initial: n = number of electrons in the conduction band
    hole = hole state below Fermi's level / head = particle state above Fermi's level / k = scattering potential"""
    nfermi = int((n + 1) / 2)
    mel = np.zeros((nfermi, nfermi))
    evali, eveci__, adm_eneri_ = dig_ham(n, 0, w)
    evalf, evecf__, adm_enerf_ = dig_ham(n, k, w)
    for bra in range(n):
        for ket in range(nfermi):
            if bra == hole:
                mel[bra][ket] = np.dot(evecf__[head], eveci__[ket])
            if bra < nfermi and bra != hole:
                mel[bra][ket] = np.dot(evecf__[bra], eveci__[ket])
    ener_excit = evalf[head] - evalf[hole]
    ener_norm = ener_excit / normaliz(n, w)
    return ener_excit, np.power(la.det(mel), 2), ener_norm


def spectrum(ni, nf, k, w, head):
    """"compute xps rates for a logarithmic sequence of energies of a fixed head"""
    n_erg: int = int((nf - ni) / 2)
    erg_ = np.zeros((n_erg))
    rate_ = np.zeros_like(erg_)
    erg_norm_ = np.zeros_like(rate_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = int((n + 1) / 2)
        n_hole = nfermi - head
        n_excit = nfermi + head - 1
        erg_[count], rate_[count], erg_norm_[count] = xps_proj(n, k, w, n_hole, n_excit)
        count += 1
    return erg_, rate_


def spectrum_sec(ni, nf, k, w, head):
    """"compute secondary xps rates for a logarithmic sequence of energies above a fixed head"""
    erg_sec_ = np.zeros((int((nf-ni)/2)*np.power(head - 1, 2)))
    rate_sec_ = np.zeros_like(erg_sec_)
    erg_sec_norm_ = np.zeros_like(erg_sec_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = int((n + 1) / 2)
        n_hole = nfermi - head
        n_excit = nfermi + head - 1
        for j in range(n_hole + 1, nfermi):
            for u in range(nfermi, n_excit):
                erg_sec_[count], rate_sec_[count], erg_sec_norm_[count] = xps_proj(n, k, w, j, u)
                count += 1
    return erg_sec_, rate_sec_


def convolution(ni, nf, k, w, head):
    """compute xps rate with a convolution with a box function"""
    erg_, rate_ = spectrum(ni, nf, k, w, head)
    erg_sec_, rate_sec_ = spectrum_sec(ni, nf, k, w, head)
    erg_conv_ = np.zeros(len(rate_)-2)
    rate_conv_ = np.zeros_like(erg_conv_)
    soma = np.zeros(len(erg_)-2)
    for i in range(len(erg_)-2):
        for j in range(len(erg_sec_)):
            if np.sqrt(erg_[i+1]*erg_[i+2]) < erg_sec_[j] < np.sqrt(erg_[i]*erg_[i+1]):
                soma[i] += rate_sec_[j]
        rate_conv_[i] = (soma[i] + rate_[i+1]) / np.power(np.log(np.power(lamb(), 2)), 1)
        erg_conv_[i] = erg_[i+1]
    return erg_conv_, rate_conv_


def binarystates(n, head):
    """List of every possible state for a number n"""
    nfermi = int((n+1)/2)
    decimal_states_ = []
    for i in range(2**(n+1)):
        nn = gmpy2.mpz(i)
        if gmpy2.popcount(nn) == nfermi:
            decimal_states_.append(nn)
    l = len(decimal_states_)
    filled__ = np.zeros((l, nfermi))
    list = []
    for j in range(l):
        count = 0
        for k in range(nfermi):
            level = gmpy2.bit_scan1(decimal_states_[j], count)
            filled__[j][k] = level
            count = level + 1
    for i in range(1, len(filled__)):
        for j in range(nfermi):
            #if j < nfermi - head:
            #    if filled__[i][j] != filled__[0][j]:
            #        list.append(i)
            if j > nfermi - head:
                if filled__[i][j] > nfermi + 2:
                    list.append(i)
    return np.delete(filled__, list, 0)


def xps_proj_binary(n, k, w, head):
    """compute all the xps ray (primary and secondary) for states in binary form"""
    m__ = binarystates(n, head)
    evali, eveci__, adm_eneri_ = dig_ham(n, 0, w)
    evalf, evecf__, adm_enerf_ = dig_ham(n, k, w)
    nfermi = int((n + 1) / 2)
    rate_ = np.zeros(len(m__)-1)
    erg_ = np.zeros(len(m__)-1)
    for state in range(1, len(m__)):
        mel = np.zeros((nfermi, nfermi))
        sum_hole = 0
        sum_excit = 0
        for bra in range(nfermi):
            for ket in range(nfermi):
                mel[bra][ket] = np.dot(evecf__[int(m__[state][bra])], eveci__[int(m__[0][ket])])
            ener_hole = evalf[int(m__[0][bra])]
            ener_excit = evalf[int(m__[state][bra])]
            sum_hole += ener_hole
            sum_excit += ener_excit
        erg_[state-1] = sum_excit - sum_hole
        rate_[state-1] = np.power(la.det(mel), 2)
    return erg_, rate_

def binary_convolution(ni, nf, k, w, head):
    """compute all the xps ray for states in binary form with a convolution box function"""
    erg_, rate_ = spectrum(ni, nf, k, w, head)
    rate_conv_ = np.zeros(len(erg_) - 2)
    erg_conv_ = np.zeros_like(rate_conv_)
    for n in range(ni, nf-4, 2):
        ergs_, rates_ = xps_proj_binary(n, k, w, head)
        for i in range(len(erg_) - 2):
            for j in range(len(ergs_) - 2):
                if np.sqrt(erg_[i + 1] * erg_[i + 2]) < ergs_[j] < np.sqrt(erg_[i] * erg_[i + 1]):
                    rate_conv_[i] += rates_[j]
                    erg_conv_[i] = erg_[i+1]
    return erg_conv_, rate_conv_ / (np.log(np.power(lamb(), 2)))


def dig_ham_imp(n, k, d, v, w):
    """Hamiltonian: n(odd) = Matrix Dimension / k = scattering potential / v = impurity bounding energy"""
    if n % 2 == 0:
        print("Sorry: n must be odd")
        exit(1)

    else:
        ham__ = np.zeros((n + 1, n + 1))
        ham__[0][0] = d
        ham__[1][1] = k
        ham__[1][2] = ham__[2][1] = T / np.sqrt(w)
        ham__[0][1] = ham__[1][0] = v
        for i in range(2, (n + 1) - 1):
            ham__[i][i + 1] = ham__[i + 1][i] = t_n(i - 2, w)
        eval_, evec__ = la.eigh(ham__)
        return eval_, evec__.T, eval_ / normaliz(n, w)


def xps_proj_imp(n, k, d, v, w, hole, excit):
    """Determinant of a projection of the final state over the initial: ne = number of electrons in the conduction band
    hole = hole state below Fermi's level / excit = particle state above Fermi's level / k = scattering potential"""
    nfermi = int((n + 1) / 2)
    mel = np.zeros((nfermi, nfermi))
    evali, eveci__, adm_eneri_ = dig_ham_imp(n, 0, d, v, w)
    evalf, evecf__, adm_enerf_ = dig_ham_imp(n, k, d, v, w)
    for bra in range(n):
        for ket in range(nfermi):
            if bra == hole:
                mel[bra][ket] = np.dot(evecf__[excit], eveci__[ket])
            if bra < nfermi and bra != hole:
                mel[bra][ket] = np.dot(evecf__[bra], eveci__[ket])
    ener_excit = evalf[excit] - evalf[hole]
    return ener_excit, np.power(la.det(mel), 2)


def spectrum_imp(ni, nf, k, d, v, w, head):
    """"compute xps rates for a logarithmic sequence of energies of a fixed head"""
    n_erg: int = int((nf - ni) / 2)
    erg_imp_ = np.zeros((n_erg))
    rate_imp_ = np.zeros_like(erg_imp_)

    count = 0
    for n in range(ni, nf, 2):
        nfermi = int((n + 1) / 2)
        n_hole = nfermi - head
        n_excit = nfermi + head - 1
        erg_imp_[count], rate_imp_[count] = xps_proj_imp(n, k, d, v, w, n_hole, n_excit)
        count += 1
    return erg_imp_, rate_imp_


def xps_proj_imp_binary(n, k, d, v, w, head):
    """compute all the xps ray (primary and secondary) for states in binary form"""
    m__ = binarystates(n, head)
    evali, eveci__, adm_eneri_ = dig_ham_imp(n, 0, d, v, w)
    evalf, evecf__, adm_enerf_ = dig_ham_imp(n, k, d, v, w)
    nfermi = int((n + 1) / 2)
    rate_imp_ = np.zeros(len(m__)-1)
    erg_imp_ = np.zeros(len(m__)-1)
    for state in range(1, len(m__)):
        mel = np.zeros((nfermi, nfermi))
        sum_hole = 0
        sum_excit = 0
        for bra in range(nfermi):
            for ket in range(nfermi):
                mel[bra][ket] = np.dot(evecf__[int(m__[state][bra])], eveci__[int(m__[0][ket])])
            ener_hole = evalf[int(m__[0][bra])]
            ener_excit = evalf[int(m__[state][bra])]
            sum_hole += ener_hole
            sum_excit += ener_excit
        erg_imp_[state-1] = sum_excit - sum_hole
        rate_imp_[state-1] = np.power(la.det(mel), 2)
    return erg_imp_, rate_imp_


def binary_convolution_imp(ni, nf, k, d, v, w, head):
    """compute all the xps ray for states in binary form with a convolution box function"""
    erg_, rate_ = spectrum_imp(ni, nf, k, d, v, w, head)
    rate_conv_ = np.zeros(len(erg_) - 2)
    erg_conv_ = np.zeros_like(rate_conv_)
    for n in range(ni, nf-4, 2):
        ergs_, rates_ = xps_proj_imp_binary(n, k, d, v, w, head)
        for i in range(len(erg_) - 2):
            for j in range(len(ergs_) - 2):
                if np.sqrt(erg_[i + 1] * erg_[i + 2]) < ergs_[j] < np.sqrt(erg_[i] * erg_[i + 1]):
                    rate_conv_[i] += rates_[j]
                    erg_conv_[i] = erg_[i+1]
    return erg_conv_, rate_conv_


def inside(ni, nf, k, d, v, head):
    es = []
    ws = []
    eps = []
    erg_, rate_ = spectrum_imp(ni, nf, k, d, v, 2, head)
    dw = 1 / np.power(10, 7)
    iw = int_w() + dw
    for i in range(len(int_epsilon())):
        for p in range(len(int_w()) - 1):
            for j in range(len(erg_)):
                if (spectrum_imp(ni, nf, k, d, v, int_w()[p], head)[0][j] - int_epsilon()[i]) * (
                        spectrum_imp(ni, nf, k, d, v, int_w()[p + 1], head)[0][j]
                        - int_epsilon()[i]) < 0:
                    es.append(j)
                    ws.append(p)
                    eps.append(int_epsilon()[i])
    erg_w_ = np.zeros(len(es))
    rate_w_ = np.zeros(len(es))
    for i in range(len(es)):
        erg_w_[i] = eps[i]
        rate_w_[i] = spectrum_imp(ni, nf, k, d, v, int_w()[ws[i]], head)[1][es[i]]
    return erg_w_, rate_w_
