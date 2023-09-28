## LIBRARIES ##
import numpy as np
import scipy.io
import utilities as ut
from sympy import Matrix, init_printing, pprint

## SELECT THE NUMBER OF SIMULATIONS TO PERFORM IN THE CONFIDENCE REGION ##
# IT TAKES ABOUT 15 seconds PER ITERATION (CTM and HAWKES together) #
# It could be necessary to close the plots to keep the code run #
M = 20

## Conventions ##
seed = 5
np.random.seed(seed)

## Load data ##
mat = scipy.io.loadmat('data.mat')
matrix = mat['matrice']
startDate = 724277
endDate = 737821

## Original version
# flag = 1
# minRating = 3

## Facultative version 1
# flag = 2
# minRating = 3

## Facultative version 2
flag = 1
minRating = 2

## 1. DATA CLEANING ##
dataset_new = ut.cleanDataset(matrix[:, [0, 1, 2, 4]], startDate, endDate, flag)

## 2. CALIBRATION ##
# Countinuous Time Markov #
qmatrix, R, N = ut.calibrationCTM(dataset_new)
qmatrix_to_print = np.trunc(100*np.copy(qmatrix)*10**3)/(10**3)
# print
init_printing()
print()
print('-- CTM Calibration --')
print()
print('Q matrix: (values in percentage)')
pprint(Matrix(qmatrix_to_print))

# Multidimensional Hawkes #
# initial guesses for Hawkes
alpha0 = 2
tau0 = 0.5
# find corporates' indices
corporates = ut.findCorporates(dataset_new)
# calibration
alphaHat, tauHat, qHat = ut.calibrationHawkes(dataset_new, corporates, alpha0, tau0, endDate, R, N, minRating)
print()
print('-- Hawkes Calibration --')
print()
print('α: ', alphaHat)
print('τ: ', tauHat)
qmatrix_to_print = np.trunc(100*np.copy(qHat)*10**3)/(10**3)
print('Q matrix: (values in percentage)')
pprint(Matrix(qmatrix_to_print))

## PLOT LOGLIKELIHOOD ##
# select the numer of points to have for each axis M (it does take a bit of time)
M_plot = 20 # INCREASE THE NUMBER TO HAVE A BETTER PLOT
alphas, betas, logL = ut.plotLogLikeHawkes(dataset_new, corporates, endDate, R, N, minRating, M_plot)

## SIMULATION ##
# Continuous Time Markov #
dataset_sim = ut.simulationCTM(dataset_new, corporates, qmatrix)
qmatrix_sim, R_sim, N_sim = ut.calibrationCTM(dataset_sim)
# print
qmatrix_to_print_sim = np.trunc(100*qmatrix_sim*10**3)/(10**3)
init_printing()
print()
print('-- CTM Calibration (simulated Dataset) --')
print()
print('Q matrix simulated: (values in percentage)')
pprint(Matrix(qmatrix_to_print_sim))

# Multidimensional Hawkes #
dataset_sim = ut.simulationHawkes(dataset_new, corporates, qHat, minRating, alphaHat, tauHat)
qmatrix_sim, R_sim, N_sim = ut.calibrationCTM(dataset_sim)
corporates_sim = ut.findCorporates(dataset_sim)
alphaHat_sim, tauHat_sim, qHat_sim = ut.calibrationHawkes(dataset_sim, corporates_sim, alpha0, tau0, endDate, R_sim, N_sim, minRating)
print()
print('-- Hawkes Calibration (simulated Dataset) --')
print()
print('α: ', alphaHat_sim)
print('τ: ', tauHat_sim)

## CONFIDENCE INTERVAL CTM ##
alpha = 0.05
interval_qCTM = ut.confidenceIntervalCTM(dataset_new, corporates, qmatrix, M, alpha)
print()
print('-- CONFIDENCE INTERVAL Q CTM --')
print()
print('Lower Q CTM: ')
qmatrix_to_print_sim = np.trunc(100*interval_qCTM[0, :, :]*10**3)/(10**3)
pprint(Matrix(qmatrix_to_print_sim))
print('Upper Q CTM: ')
qmatrix_to_print_sim = np.trunc(100*interval_qCTM[1, :, :]*10**3)/(10**3)
pprint(Matrix(qmatrix_to_print_sim))

## CONFIDENCE REGION HAWKES ##
corporates= ut.findCorporates(dataset_new)
alphaHat_sim, tauHat_sim, qHat_sim, interval_qHwk =ut.confidenceRegionHawkes(dataset_new, corporates, alphaHat, tauHat, qHat, endDate, minRating, M, alpha)
print()
print('-- CONFIDENCE INTERVAL Q HAWKES --')
print()
print('Lower Q Hawkes: ')
qmatrix_to_print_sim = np.trunc(100*interval_qHwk[0, :, :]*10**3)/(10**3)
pprint(Matrix(qmatrix_to_print_sim))
print('Upper Q Hawkes: ')
qmatrix_to_print_sim = np.trunc(100*interval_qHwk[1, :, :]*10**3)/(10**3)
pprint(Matrix(qmatrix_to_print_sim))
print()

# NORMALITY CHECK #
ut.normalityCheck(alphaHat_sim.flatten(), tauHat_sim.flatten())

# CONFIDENCE REGION PLOT #
ut.plotConfidenceRegion(alphaHat_sim.flatten(), tauHat_sim.flatten(), alpha, M)