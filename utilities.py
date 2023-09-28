import numpy as np
import math as mt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
import scipy.optimize
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.stats import kstest, probplot, norm, chi2, t
import pylab
import pingouin as pg

########################################################################################################################
########################################################################################################################


def cleanDataset(dataset, startdate, enddate, flag):
    '''
    function executing the following commands for cleaning the dataset:
    -replacing 'NaN' with enddate
    -selection of the observations between startdate and enddate
    -aggregation of notches in the new rating system
      1)	1
      2)	2 3 4
      3)	5 6 7
      4)	8 9 10
      5)	11 12 13
      6)	14 15 16
      7)	17 18 19	20 21
      8)	23
      WR  24
    -handle the WR as written in Baviera Note

    INPUTS
    dataset: original DRD Moody's Dataset
    startDate: initial date of the analysis
    endDate: final date of the analysis
    flag: 1(default not considered as a pure absorbing state)
          2(default considered as a pure absorbing state)


    OUTPUTS
    dataset_cleaned

    FUNCTION USED
    mergeRows
    '''

    # Replace NaN with enddate
    dataset[np.isnan(dataset[:, 2]), 2] = enddate

    # Select observations inside the considered time interval
    dataset = dataset[(dataset[:, 1] < enddate) &
                      (dataset[:, 2] > startdate), :]

    # Replace obs initial date with startdate when initial date is anterior to startdate
    dataset[dataset[:, 1] < startdate, 1] = startdate

    # Replace obs final date with enddate when final date is posterior to enddate
    dataset[dataset[:, 2] > enddate, 2] = enddate

    # Replace ratings with new rating system
    dataset[dataset[:, -1] == 22, -1] = 24

    # Aggregation of notches into new 7+1(D) classes
    idxR = [0, 1, 4, 7, 10, 13, 16, 21, 23]  # indexes of ratings intervals
    for i in range(len(idxR)-1):
        idx = (dataset[:, -1] > idxR[i]) & (dataset[:, -1] <= idxR[i + 1])  # indexes of obs inside the i-th interval
        dataset[idx, -1] = i+1  # replace with i-th rating

    # Handle D and WR wrt different situations
    if flag == 1:  # D is not an absorbing state
        Default = (dataset[:, -1] == 8)  # D
        dataset[:, 0] = dataset[:, 0] + np.cumsum(np.concatenate([[False], Default[:-1]]))  # Reentry after default as an independent issuer
        dataset[Default, 2] = enddate  # extend final date of default to enddate
    elif flag == 2:  # D is an absorbing state
        corporates = np.unique(dataset[:, 0])  # vector of idnumber of each corporate reported one time and sorted
        dataset_new = []
        for c in corporates:
            datasetCorporate = dataset[dataset[:, 0] == c, :]  # take the dataset just for the company c
            if np.sum(datasetCorporate[:, -1] == 8) == 0:
                dataset_new.append(datasetCorporate)
            else:
                firstDefault = np.where(datasetCorporate[:, -1] == 8)[0][0]  # index of the first default of a corporate
                datasetCorporate[firstDefault, 2] = enddate
                dataset_new.append(datasetCorporate[:firstDefault + 1, :])
        dataset = np.concatenate(dataset_new, axis=0)

    # WR as first observation of the corporate
    to_cancel = (dataset[1:, -1] == 24) & (dataset[:-1, 0] != dataset[1:, 0])  # first obs of the issuer, i.e. different issuer number
    idx = np.where(to_cancel)
    np.sum(to_cancel)  # number cancelled
    b=np.sum(to_cancel)
    dataset = np.delete(dataset, np.array(idx) + 1, axis=0)
    # np.clear(idx)

    # WR as last observation of the corporate
    to_cancel = (dataset[:-1, -1] == 24) & (dataset[:-1, 0] != dataset[1:, 0])  # last obs of the issuer, i.e. different issuer number
    idx = np.where(to_cancel)
    c=np.sum(to_cancel)  # number cancelled
    dataset = np.delete(dataset, np.array(idx), axis=0)

    # Condition 1 Baviera
    nDay = 365
    to_adjust = (dataset[:-1, 0] == dataset[1:, 0]) & (dataset[:-1, -1] == 24) & (dataset[1:, -1] == 8) & (
            dataset[:-1, 2] - dataset[:-1, 1] < nDay)  # less than nYears later
    idx = np.where(to_adjust)
    a= np.sum(to_adjust)
    if a!=0:
        dataset[np.concatenate(([False], to_adjust[:-1])), 1] = dataset[to_adjust, 1]
        dataset = np.delete(dataset, np.array(idx), axis=0)

    # Condition 2 Baviera
    to_adjust = (dataset[:-1, 0] == dataset[1:, 0]) & (dataset[:-1, -1] == 24) & (dataset[1:, -1] == 8) & (dataset[:-1, 2] - dataset[:-1, 1] >= nDay)
    a=np.sum(to_adjust)
    to_adjust= np.array(to_adjust)
    idx = np.concatenate((np.array(np.where(to_adjust)) + 1, np.array(np.where(to_adjust))))
    # np.sum(to_adjust)
    dataset = np.delete(dataset, np.array(idx), axis=0)  # cancel WR & D

    # D as first obs of the corporate
    to_cancel = (dataset[1:, 3] == 8) & (dataset[:-1, 0] != dataset[1:, 0])  # first obs of the issuer, i.e. different issuer number
    idx = np.where(to_cancel)
    np.sum(to_cancel)
    dataset = np.delete(dataset, np.array(idx) + 1, axis=0)

    # Condition 3 Baviera
    to_postpone = (dataset[1:-1, -1] == 24) & (dataset[1:-1, 2] - dataset[1:-1, 1] <= nDay) & (
            dataset[1:-1, 0] == dataset[2:, 0]) & (dataset[:-2, 0] == dataset[1:-1, 0])
    idx = np.where(to_postpone)
    idx=np.array(idx)
    np.sum(to_postpone)
    dataset[idx, 2] = dataset[idx + 2, 1]
    dataset = np.delete(dataset, np.array(idx) + 1, axis=0)

    # Condition 4 Baviera
    to_delete = (dataset[:, -1] == 24) & (dataset[:, 2] - dataset[:, 1] > nDay)
    idx = np.where(to_delete)
    np.sum(to_delete)  # number changed issuer
    dataset[:, 0] = dataset[:, 0] + np.cumsum(np.concatenate(([False], to_delete[:-1])))
    dataset = np.delete(dataset, np.array(idx), axis=0)

    # Merge consecutive obs with the same rating
    corporates = np.unique(dataset[:, 0])
    dataset_merged = np.empty((0, dataset.shape[1]))
    for c in corporates:
        dataset_corporate = dataset[dataset[:, 0] == c, :]
        dataset_merged = mergeRows(dataset_corporate, dataset_merged)

    return dataset_merged

########################################################################################################################
########################################################################################################################


def mergeRows(dataset_corporate, dataset_merged):
    '''
    function that aggregates same cosnecutive rating after the change of the
    rating evaluation

    INPUT
    datasetCorporate:dataset of the corporate c
    dataset_merged: dataset where consecutive rows with same rating

    OUTPUT
    dataset_out: dataset merged
    '''
    tmp_mat = np.array([dataset_corporate[0, :]])

    for ii in range(1, len(dataset_corporate[:, 0])):
        if dataset_corporate[ii, -1] == dataset_corporate[ii - 1, -1]:
            tmp_mat[-1, 2] = dataset_corporate[ii, 2]
        else:
            tmp_mat = np.concatenate((tmp_mat, np.array([dataset_corporate[ii, :]])), axis=0)

    dataset_out = np.concatenate((dataset_merged, tmp_mat), axis=0)

    return dataset_out

########################################################################################################################
########################################################################################################################


def computeIic(rating_i, datasetCorporate, minRating, T, TTI):
    '''
    Function that computes the first 3 dimensions of Itimes for jumps from i
    for corporate c


    INPUTS
    i: initial rating at jump-time
    datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
    minRating: minimum rating from which there is the downgrade momentum effect
    T: end date
    TT: auxiliary preallocated 2-dimensional matrix


    OUTPUT
    TT: filled matrix with all the sets of consecutive downgrades of interest
        both for 1st and 2nd layer
    '''

    TT = np.nan*np.copy(TTI)  # init
    indexesi = np.argwhere(datasetCorporate[:, -1] == rating_i)  # indexes of i - th rating for the corporate c
    indexesi = indexesi[indexesi > 0]  # keep all the indexes except if it is the first element
    for i in range(len(indexesi)):
        minR = datasetCorporate[0:indexesi[i]+1, -1] > minRating  # boolean vector to check if ratings are greater than minRating
        is_downgrade = minR[:-1]*(np.diff(datasetCorporate[:indexesi[i]+1, -1]) > 0)
        if not not len(np.argwhere(is_downgrade == 0)):  # if it is nonempty
            idxstart = int(np.argwhere(is_downgrade == 0)[-1] + 1)
        else:
            idxstart = 0
        if idxstart < indexesi[i]:
            dt1 = (datasetCorporate[indexesi[i], 1] - datasetCorporate[(idxstart + 1):indexesi[i]+1, 1]) / 365
            dt2 = (min(datasetCorporate[indexesi[i], 2], T) - datasetCorporate[(idxstart + 1):indexesi[i]+1, 1]) / 365
            TT[:len(dt1), i, 0] = dt1
            TT[:len(dt2), i, 1] = dt2

    return TT

########################################################################################################################
########################################################################################################################


def computeJijc(rating_i, j, datasetCorporate, minRating, TTJ):
    '''
    Function that computes the first 2 dimensions of Jtimes for jumps from i
    to j for corporate c


    INPUTS
    i: initial rating at jump-time
    J: arrival rating at jump-time
    datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
    minRating: minimum rating from which there is the downgrade momentum effect
    TT: auxiliary preallocated 2-dimensional matrix


    OUTPUT
    TT: filled matrix with all the sets of consecutive downgrades of interest
    '''

    TT = np.nan * np.copy(TTJ)  # init

    if any((datasetCorporate[:, -1] == rating_i) == 1):
        indexesi = np.argwhere((datasetCorporate[:-1, -1] == rating_i) * (datasetCorporate[1:, -1] == j)) # indexes of i - th rating for the corporate c
        indexesi = indexesi[indexesi > 0]
        for i in range(len(indexesi)):
            minR = datasetCorporate[0:indexesi[i]+1, -1] > minRating  # boolean vector to check if ratings are greater than minRating
            is_downgrade = minR[:-1] * (np.diff(datasetCorporate[:indexesi[i]+1, -1]) > 0)
            if not not len(np.argwhere(is_downgrade == 0)):  # if it is nonempty
                idxstart = int(np.argwhere(is_downgrade == 0)[-1] + 1)
            else:
                idxstart = 0

            if idxstart < indexesi[i]:
                dt = (datasetCorporate[indexesi[i], 2] - datasetCorporate[idxstart:indexesi[i], 2]) / 365
                TT[:len(dt), i] = dt

    return TT

########################################################################################################################
########################################################################################################################


def helpParJ(i, datasetCorporate, minRating, J3d, TT):
    '''
    The function returns the Δt matrix for the i-th starting rating, for
    every j>i arrival rating for input corporate c


    INPUTS
    i: initial rating at jump-time
    datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
    minRating: minimum rating from which there is the downgrade momentum effect
    J3d: J matrix of 3 dimensions
    TT: auxiliary preallocated 2-dimensional matrix


    OUTPUT
    J: Δt matrix for J up to first 3 dimensions for the corporate c


    FUNCTION USED
    computeJijc
    '''

    J = np.nan * np.copy(J3d)  # init

    # for j in range(i+1,9):
    for j in range(i+1, 9):
        J[:, :, j-i-1] = computeJijc(i, j, datasetCorporate, minRating, TT) # store data for the c corporate for every j-th arrival rating
    return J

########################################################################################################################
########################################################################################################################


def SemiLogLikeHawkes(dataset, corporates, minRating, T, dim1, dim2):
    '''
    Function that conmputes the Δtimes for I and J for the LogLikelihood computation


    INPUTS
    dataset: cleaned DRD Moody's Dataset
    corporates: matrix having in the first row the list of all the c ∈ C
                              in the second row the first index of c in the dataset
                              in the third row the last index of c in the dataset
    minRating: minimum rating from which there is the downgrade momentum effect
    T: end Date of analysis
    dim1: size of dimension 1 of I and J matrices
    dim2: size of dimension 2 of I and J matrices


    OUTPUTS
    Itimes: 5-dimensional matrix with Δtimes for I for the LogLikelihood computation
    Jtimes: 5-dimensional matrix with Δtimes for J for the LogLikelihood computation
        The matrices dimensions are as follows:
        Itimes: dim1: consecutive downgrades
                dim2: sets of consecutive downgrades
                dim3: 1st layer for Δt1 and 2nd layer for Δt2
                dim4: full corporates C set
                dim5: i start rating (from minRating+1 to 7)
        Jtimes: dim1: consecutive downgrades
                dim2: sets of consecutive downgrades
                dim3: j arrival ratings
                dim4: full corporates C set
                dim5: i start rating (from minRating+1 to 7)


    FUNCTIONS USED
    helpParJ
    computeIic
    '''

    # preallocating spaces for matrices (preallocating for speed)
    lenCo = max(np.shape(corporates)) # number of corporates
    Itimes = np.nan * np.ones([dim1, dim2, 2, lenCo, 7 - minRating]) # matrix of times for I
    Jtimes = np.nan * np.ones([dim1, dim2, 8 - minRating - 1, lenCo, 7 - minRating]) # matrix of times for J
    TTI = np.nan * np.ones([dim1, dim2, 2]) # auxiliary matrix for I computation
    TTJ = np.nan * np.ones([dim1, dim2]) # auxiliary matrix for J computation
    IvecC = np.nan * np.ones([dim1, dim2, 2, lenCo]) # auxiliary matrix for corporates data for I computation
    JmatC = np.nan * np.ones([dim1, dim2, 8 - minRating - 1, lenCo]) # auxiliary matrix for corporates and j-th rating sum data for j computation
    J3d = np.nan * np.ones([dim1, dim2, 8 - minRating - 1]) # auxiliary first 3 dimensions of J vector of times

    # loop on i-th starting ratings for LogLikelihood computation
    for i in range(minRating+2, 8):
        # init
        IvecC = np.nan * IvecC
        JmatC = np.nan * JmatC
        for iii in range(lenCo): # loop on corporates c ∈ C
            datasetCorporate = dataset[corporates[1, iii]:corporates[2, iii]+1, :] # select dataset relative to the corporate[iii]
            JmatC[:, :, :, iii] = helpParJ(i, datasetCorporate, minRating, J3d, TTJ) # store data for J for each company
            IvecC[:, :, :, iii] = computeIic(i, datasetCorporate, minRating, T, TTI) # store data for I for each company

        Jtimes[:, :, :, :, i-minRating-1] = JmatC # add the data to the matrix
        Itimes[:, :, :, :, i-minRating-1] = IvecC # add the data to the matrix
    # Matrix dimension reduction
    # In order to improve fmincon/fminsearch speed we reduce the matrixes
    # dimensions, so that basic operations are much faster

    # init
    ItimesNew = np.nan * np.copy(Itimes)
    JtimesNew = np.nan * np.copy(Jtimes)
    lenI = np.zeros([7-minRating, 1])
    lenJ = np.zeros([7-minRating, 1])

    for d5 in range(7-minRating): # for loop on 5-th dimension
        # find submatrixes of dt where there isn't any NaN (find corporates with useful data)
        idxI = np.any(~np.isnan(Itimes[:, :, :, :, d5]), axis=(0, 1, 2))
        idxJ = np.any(~np.isnan(Jtimes[:, :, :, :, d5]), axis=(0, 1, 2))
        # store the clean data in a temp matrix
        tmpI = Itimes[:, :, :, idxI, d5]
        tmpJ = Jtimes[:, :, :, idxJ, d5]
        # store the number of corporates with useful data (different from all NaNs)
        lenI[d5] = tmpI.shape[3]
        lenJ[d5] = tmpJ.shape[3]
        # finally store the clean data in the new matrices
        ItimesNew[:, :, :, :int(lenI[d5]), int(d5)] = tmpI
        JtimesNew[:, :, :, :int(lenJ[d5]), int(d5)] = tmpJ

    # cut the final matrices to the max number of useful corporates
    return ItimesNew[:, :, :, :int(max(lenI)), :], JtimesNew[:, :, :, :int(max(lenJ)), :]

########################################################################################################################
########################################################################################################################


def qMatrixHawkes(Itimes, alpha, tau, R, Rvec, N, minRating):
    '''
    Function that computes Q matrix following a Multidimensional Hawkes Model for credit rating migrations

    INPUTS
    Itimes: 5-dimensional matrix with Δtimes for I for the LogLikelihood computation:
            dim1: consecutive downgrades
            dim2: sets of consecutive downgrades
            dim3: 1st layer for Δt1 and 2nd layer for Δt2
            dim4: full corporates C set
            dim5: i start rating (from minRating+1 to 7)
    alpha: calibrated α parameter
    tau: calibrated τ parameter
    R: R vector from CTM calibration
    Rvec: R vector last components dimension-adjusted
    N: N vector from CTM calibration
    minRating: minimum rating from which there is the downgrade momentum effect

    OUTPUTS
    qHat: calibrated Q matrix
    '''

    qHat = np.divide(N, R[:, np.newaxis])  # init with qHat = qmatrix from CTM

    # denominator as function (handle):
    # Denominator_i = R + αI
    def Denominator_i(alpha, tau, ith):
        #return Rvec[ith+1] + alpha * np.nansum(tau * np.nansum(np.exp(-Itimes[:, :, 0, :, ith] / tau) - np.exp(-Itimes[:, :, 1, :, ith] / tau),axis=(0, 1)),axis=0)
        return Rvec[ith] + alpha * np.nansum(tau * np.nansum(np.exp(-Itimes[:, :, 0, :, ith] / tau) - np.exp(-Itimes[:, :, 1, :, ith] / tau), axis=(0, 1)), axis=0)
    for i in range(minRating, 7):  # loop on i-th starting ratings
        denominator = Denominator_i(alpha, tau, i-minRating)
        qHat[i, i + 1:] = N[i, i + 1:] / denominator

    return qHat

########################################################################################################################
########################################################################################################################


def calibrationHawkes(dataset, corporates, alpha0, tau0, T, R, N, minRating):
    '''
    Function that calibrates a Multidimensional Hawkes Model for credit rating migrations

    INPUTS
    dataset: cleaned DRD Moody's Dataset
    corporates: matrix having in the first row the list of all the c ∈ C
                              in the second row the first index of c in the dataset
                              in the third row the last index of c in the dataset
    alpha0: starting guess for α parameter calibration
    tau0: starting guess for τ parameter calibration
    T: end Date of analysis
    R: R vector from CTM calibration
    N: N vector from CTM calibration
    minRating: minimum rating from which there is the downgrade momentum effect

    OUTPUTS
    alphaHat: calibrated α parameter
    tauHat: calibrated τ parameter
    qHat: calibrated Q matrix

    FUNCTIONS USED
    SemiLogLikeHawkes
    qMatrixHawkes
    '''

    # init
    dim1 = 8-minRating-1  # dimension 1 matrix
    dim2 = int(np.floor(np.max(corporates[2, :] - corporates[1, :])) / 2) + 1  # dimension 2 matrix

    # SemiLoghLikelihood computation (only the part that depends from α and τ)
    # Itimes, Jtimes: matrices of Δtimes to be plugged at the exponentials in the formula
    Itimes, Jtimes = SemiLogLikeHawkes(dataset, corporates, minRating, T, dim1, dim2)

    # Dimensions adjustment of N and R for future computation
    Nvec = np.zeros(7 - minRating)
    for i in range(minRating, 7):
        Nvec[i - minRating] = np.sum(N[i, i + 1:])

    Rvec = R[minRating:]

    # Funcions (handle) for first (I) and second (J) components of the SemiLoghLikelihood
    def Icomp(alpha, tau):
        return np.nansum(-Nvec * np.log(Rvec + alpha * np.nansum(tau * np.nansum(np.exp(-Itimes[:, :, 0, :, :] / tau) - np.exp(-Itimes[:, :, 1, :, :] / tau), axis=(0, 1)), axis=0)), axis=None)

    def Jcomp(alpha, tau):
        return np.nansum(np.log(1 + alpha * np.nansum(np.exp(-Jtimes / tau), axis=0)), axis=None)
    # function to minimize for finding the optimal parameters
    def LogLiketoMax(param):
        return -(Jcomp(param[0], param[1]) + Icomp(param[0], param[1]))

    param0 = np.array([alpha0, tau0])  # initial guess
    options = {'disp': False}  # minimize options

    # Optimal parameters finding
    res = minimize(LogLiketoMax, param0, method='Nelder-Mead', options=options)
    param_opt = res.x

    alphaHat = param_opt[0]
    tauHat = param_opt[1]

    # Q matrix computation
    qHat = qMatrixHawkes(Itimes, alphaHat, tauHat, R, Rvec, N, minRating)

    return alphaHat, tauHat, qHat

########################################################################################################################
########################################################################################################################


def simulationHawkes(dataset, corporates, qmatrix, minRating, alpha, tau):
    '''
    Function that simulates a new dataset following a Multidimensional Hawkes Process

    INPUTS
    dataset: cleaned DRD Moody's Dataset
    corporates: matrix having in the first row the list of all the c ∈ C
                              in the second row the first index of c in the dataset
                              in the third row the last index of c in the dataset
    qmatrix: Q matrix from Hawkes calibration
    minRating: minimum rating from which there is the downgrade momentum effect
    alpha: α parameter from Hawkes calibration
    tau: τ parameter from Hawkes calibration

    OUTPUTS
    dataset_sim: simulated dataset according to a Multidimensional Hawkes Process
    '''

    # initialization of simulated dataset
    dataset_sim = np.nan * np.ones((int(55e3), 4))
    datasetCorporate_sim = np.nan * np.ones((20, 4))
    totcount = 0
    ratings = np.arange(1, 9)  # set of ratings

    # loop for replicating all the corporates c ∈ C
    for ii in range(corporates.shape[1]):
        c = corporates[0, ii]  # corporate c
        datasetCorporate = dataset[corporates[1, ii]:corporates[2, ii]+1, :]  # take the dataset just for the company c
        tStart = datasetCorporate[0, 1]  # start date of the simulation
        tEnd = datasetCorporate[-1, 2]  # end date of the simulation
        ratingStart = datasetCorporate[0, -1]  # initial rating

        # initialization of Simulation Algorithm
        rating_i = int(ratingStart)  # starting rate
        s = tStart  # starting time
        flag = 0  # reset the flag = 0
        datasetCorporate_sim = np.nan * datasetCorporate_sim
        count = -1  # reset of simulated datasetCorporate for c

        # Simulation Algorithm
        while s < tEnd:  # while jump-times are within the simulation period
            # If it is the first iteration or it is NOT a downgrade of interest AND NOT default
            if np.logical_and(np.logical_or(np.all(np.isnan(datasetCorporate_sim)), flag == 1), rating_i != 8):
                flag = 0  # reset the flag
                intensity = np.sum(qmatrix[rating_i-1, :])  # intensities λ = Σq, i ≠ j
                dt = -1 / intensity * np.log(np.random.uniform(0, 1))  # Δt interarrival time as exponential distribution
                s1 = s + int(dt * 365)  # update time s
                if s1 >= tEnd:  # the jump-time is after tEnd
                    # extend i-th rating up to tEnd
                    rowDataset = [c, s, tEnd, rating_i]
                    count += 1  # update the counter
                    datasetCorporate_sim[count, :] = rowDataset  # adding the row to the dataset
                    break  # exit the while loop
                else:  # the jump-time is valid
                    u = intensity * np.random.uniform(0, 1, 1)  # I draw an Uniform r.v. between [0,λ]
                    # find the arrival rating
                    segment = intensity - np.cumsum(qmatrix[rating_i-1, :])  # λ - Σ(k)q
                    rating_j = np.argmax(u >= segment)+1  # find k s.t. U ≥ λ - Σ(k)q
                    # add the jump to the dataset
                    rowDataset = [c, s, s1, rating_i]
                    count += 1  # update the counter
                    datasetCorporate_sim[count, :] = rowDataset  # adding the row to the dataset
                    rating_i = int(rating_j)  # update rating i
                    s = s1  # update time s = s1
                    continue

            # Check for Default
            elif rating_i == 8:  # If the rating is Default
                # add the jump to the dataset
                rowDataset = [c, s, tEnd, rating_i]  # extend D=8 up to tEnd
                count += 1  # update the counter
                datasetCorporate_sim[count, :] = rowDataset  # adding the row to the dataset
                break  # exit the while loop

            # From the second iteration on, check if the last time-jump was a downgrade
            # if it is not the first iteration AND it is an "interesting" downgrade
            elif rating_i > datasetCorporate_sim[count, -1] and datasetCorporate_sim[count, -1] > minRating:
                # 1) Compute rating downgrade momentum
                # find consecutive downgrades of interest:
                ratingsCorporate = np.int_(np.concatenate((datasetCorporate_sim[~np.isnan(datasetCorporate_sim[:, -1]), -1], np.array([rating_i]))))
                is_downgrade = np.logical_and(np.diff(ratingsCorporate) > 0, ratingsCorporate[:-1] > minRating)

                if not not len(np.argwhere(is_downgrade == 0)):  # if it is nonempty
                    idxstart = int(np.argwhere(is_downgrade == 0)[-1] + 1)
                else:
                    idxstart = 0

                # φ function for λijc
                tj = datasetCorporate_sim[idxstart:count+1, 2]
                ti = s
                def phi(t):
                    return np.sum(np.exp(-(t - tj) / (tau * 365)))  # φ function Σexp(-Δt/τ) rating momentum

                # A) Generate candidate jump with exponential distribution
                u = np.random.uniform(0, 1, 1)
                # fun: log(u) + ∫(q + Iqα(φ(x)))
                def fun(x):
                    return np.log(u) + np.sum((qmatrix[rating_i-1, :] * (x - ti) / 365) + ((ratings > rating_i) * (qmatrix[rating_i-1, :]) * alpha * tau * (-phi(x) + phi(ti))), axis=None)

                # eqn: log(u) + ∫(q + Iqα(φ(x))) == 0 in order to find jump-time x
                x = root_scalar(fun, bracket=[7e05, 8e05], method='brentq')  # update time x
                s1 = int(x.root)  # update time s1
                if s1 >= tEnd:  # the jump-time is after tEnd
                    # extend i-th rating up to tEnd
                    rowDataset = [c, s, tEnd, rating_i]
                    count += 1  # update the counter
                    datasetCorporate_sim[count, :] = rowDataset  # adding the row to the dataset
                    break  # exit the while loop
                else:
                    # B) draw a Uniform r.v. between [0,λ]
                    intensities_ij = qmatrix[rating_i-1, :] * (1 + np.dot(ratings > rating_i, alpha * phi(s1)))  # λ = q + IqαΣexp(-Δt/τ)
                    intensity = np.sum(intensities_ij)  # intensities λ = Σλ, i ≠ j
                    u = intensity * np.random.uniform(0, 1, 1)  # I draw a Uniform r.v. between [0,λ]
                    # find the arrival rating
                    segment = intensity - np.cumsum(intensities_ij)  # λ - Σ(k)λ
                    rating_j = np.argmax(u >= segment)+1  # find k s.t. U ≥ λ - Σ(k)λ
                    # add the jump to the dataset
                    rowDataset = [c, s, s1, rating_i]
                    count += 1  # update the counter
                    datasetCorporate_sim[count, :] = rowDataset  # adding the row to the dataset
                    rating_i = int(rating_j)  # update rating i
                    s = s1  # update time s = s1
            else:  # If it is NOT a downgrade of interest, do as in step 1)
                flag = 1

        dataset_sim[totcount: totcount + count + 1, :] = datasetCorporate_sim[~np.isnan(datasetCorporate_sim[:, 0]), :]
        totcount = totcount + count + 1  # update the counter
    dataset_sim = dataset_sim[~np.isnan(dataset_sim[:, 0]), :]  # just take the elements that are ≠ NaN
    return dataset_sim

########################################################################################################################
########################################################################################################################


def plotLogLikeHawkes(dataset, corporates, T, R, N, minRating, M):
    '''
    Function that plots the LogLikelihood of a Multidimensional Hawkes Model for credit rating migrations


    INPUTS
    dataset: cleaned DRD Moody's Dataset
    corporates: matrix having in the first row the list of all the c ∈ C
                              in the second row the first index of c in the dataset
                              in the third row the last index of c in the dataset
    T: end Date of analysis
    R: R vector from CTM calibration
    N: N vector from CTM calibration
    minRating: minimum rating from which there is the downgrade momentum effect
    M: number of points in each plot's axis


    OUTPUTS
    alphas: α points in the plot
    betas: β points in the plot
    logL: LogLikelihood grid


    FUNCTIONS USED
    SemiLogLikeHawkes
    '''
    # init
    dim1 = 8-minRating-1  # dimension 1 matrix
    dim2 = int(np.floor(np.max(corporates[2, :] - corporates[1, :])) / 2) + 1  # dimension 2 matrix

    # SemiLoghLikelihood computation (only the part that depends from α and τ)
    # Itimes, Jtimes: matrices of Δtimes to be plugged at the exponentials in the formula
    Itimes, Jtimes = SemiLogLikeHawkes(dataset, corporates, minRating, T, dim1, dim2)

    # Dimensions adjustment of N and R for future computation
    Nvec = np.zeros(7 - minRating)
    for i in range(minRating, 7):
        Nvec[i - minRating] = np.sum(N[i, i + 1:])

    Rvec = R[minRating:]

    # Funcions for first (I) and second (J) components of the SemiLoghLikelihood
    def Icomp(alpha, tau):
        return np.nansum(-Nvec * np.log(Rvec + alpha * np.nansum(
            tau * np.nansum(np.exp(-Itimes[:, :, 0, :, :] / tau) - np.exp(-Itimes[:, :, 1, :, :] / tau), axis=(0, 1)),
            axis=0)), axis=None)

    def Jcomp(alpha, tau):
        return np.nansum(np.log(1 + alpha * np.nansum(np.exp(-Jtimes / tau), axis=0)), axis=None)

    # function to minimize for finding the optimal parameters
    def LogLike(param):
        return Jcomp(param[0], param[1]) + Icomp(param[0], param[1])

    # auxiliary computations
    tempN = N.copy()
    tempN[tempN == 0] = 1

    # init
    alphas = np.linspace(2, 6, M)
    taus = np.linspace(0.2, 1, M)
    betas = 1 / taus
    reshaped_R = np.reshape(R, (7, 1))  # Reshape R to have the shape (7, 1)
    a = N * np.log(reshaped_R)  # Perform element-wise multiplication

    # LogLikelihood part 1 (independent from α and β)
    log1 = -dataset.shape[0] + np.sum(N * np.log(tempN)) - np.sum(a[:minRating, :minRating])

    # LogLikelihood part 2 (function of α and β)
    log2 = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            log2[j, k] = LogLike([alphas[k], 1 / betas[j]])

    logL = log1 + log2

    # Plot surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)
    ax.plot_surface(X, Y, logL, cmap='jet')
    ax.set_title('LogLikelihood L(α,β)', fontsize=16)
    ax.set_xlabel('α', fontsize=13)
    ax.set_ylabel('β', fontsize=13)
    ax.set_zlabel('LogLikelihood', fontsize=13)
    plt.show()

    return alphas, betas, logL

########################################################################################################################
########################################################################################################################


def simulationCTM(dataset, corporates, qmatrix):
    """
    Function simulating a fake dataset with the same corporates, start rating, and
    period of observations as the real dataset. It applies the Algorithm described in Baviera 4.1.

    Parameters:
    - dataset: cleaned DRD Moody's Dataset
    - corporates: matrix of corporates details
    - qmatrix: matrix of transition probabilities

    Returns:
    - dataset_sim: simulated dataset
    """

    # initialization of simulated dataset
    dataset_sim = np.nan * np.ones((int(55e3), 4))
    datasetCorporate_sim = np.nan * np.ones((20, 4))
    totcount = 0

    # Loop for replicating all the corporates c ∈ C
    for ii in range(corporates.shape[1]):
        c = corporates[0, ii]  # corporate c
        datasetCorporate = dataset[corporates[1, ii]:corporates[2, ii]+1, :]  # take the dataset just for the company c
        tStart = datasetCorporate[0, 1]  # start date of the simulation
        tEnd = datasetCorporate[-1, 2]  # end date of the simulation
        ratingStart = datasetCorporate[0, -1]  # initial rating

        # Initialization of Simulation Algorithm
        rating_i = int(ratingStart) # starting rate
        s = tStart  # starting time
        datasetCorporate_sim = np.nan * datasetCorporate_sim
        count = 0

        # Simulation Algorithm
        while s < tEnd:
            if rating_i == 8:  # If the rating is Default
                # Add the jump to the dataset
                rowDataset = np.array([c, s, tEnd, rating_i])
                count += 1
                datasetCorporate_sim[count, :] = rowDataset
                break

            else:
                intensity = np.sum(qmatrix[rating_i-1, :])  # Intensities λ = Σq, i ≠ j
                dt = -1 / intensity * np.log(np.random.uniform(0, 1))  # Δt interarrival time as exponential distribution
                s1 = s + int(dt * 365)  # Update time s
                if s1 >= tEnd:  # The jump-time is after tEnd
                    # Extend i-th rating up to tEnd
                    rowDataset = [c, s, tEnd, rating_i]
                    count += 1
                    datasetCorporate_sim[count, :] = rowDataset
                    break

                else:  # The jump-time is valid
                    u = intensity * np.random.uniform(0, 1, 1)  # I draw an Uniform r.v. between [0,λ]
                    # find the arrival rating
                    segment = intensity - np.cumsum(qmatrix[rating_i-1, :])  # λ - Σ(k)q
                    rating_j = np.argmax(u >= segment)+1  # Find k s.t. U ≥ λ - Σ(k)q
                    # Add the jump to the dataset
                    rowDataset = ([c, s, s1, rating_i])
                    count += 1
                    datasetCorporate_sim[count, :] = rowDataset
                    rating_i = int(rating_j)  # Update rating i
                    s = s1  # Update time s = s1
                    continue

        # Add the simulated datasetCorporate to the whole dataset
        dataset_sim[totcount+1:totcount + count+1, :] = datasetCorporate_sim[~np.isnan(datasetCorporate_sim[:, 0]), :]
        totcount += count
    dataset_sim = dataset_sim[~np.isnan(dataset_sim[:, 0]), :]  # Take the elements that are not NaN
    return dataset_sim

########################################################################################################################
########################################################################################################################


def calibrationCTM(dataset):
    '''
    function computing the estimated transition probabilities for a Continuous Time Markov Model

    INPUTS
    dataset: cleaned DRD Moody's Dataset

    OUTPUTS
    qmatrix: transition probability matrix
    R: total time up to T that all corporates spend with a rting i
    N: total number of transitions (ij) up to T by all corporates
    '''

    deltaTvec = (dataset[:, 2] - dataset[:, 1]) / 365
    ratings = (0, 1, 2, 3, 4, 5, 6, 7)
    R = np.ones(7)
    #print(R)
    N = np.zeros((7, 8))
    qmatrix = np.zeros((7, 8))
    arrivalRating = np.append(dataset[1:, -1], 0)
    tmpChangeCindex = np.argwhere((dataset[1:, 0] - dataset[:-1, 0]) != 0) # find the index of the last rating for each corporate
    arrivalRating[tmpChangeCindex] = 0

    for r in range(7):
        indexes = np.argwhere(dataset[:, -1] == r+1)
        R[r] = np.sum(deltaTvec[indexes])
        for rating in range(8):
            N[r, rating] = np.sum(arrivalRating[indexes] == rating+1)
            qmatrix[r,rating] = N[r,rating] / R[r]
    return qmatrix, R, N

########################################################################################################################
########################################################################################################################


def findCorporates(dataset):
    '''
    The function finds all the corporates in the dataset and returns a matrix 3xNcorporates.
    corporates(1,:) = corporate names (indexed numbers)
    corporates(2,:) = first indexes of dataset for each company
    corporates(3,:) = last indexes of dataset for each company
    '''

    maxCorporate = np.max(dataset[:, 0])  # last corporate for loop
    tmp = np.arange(1, maxCorporate+1)

    # corporates = np.zeros((3, tmp.size))
    a=np.isin(tmp, dataset[:, 0])
    b=tmp[a]
    corporates = np.zeros((3, b.size))
    corporates[0, :] = b  # find all corporates

    for iii in range(corporates.shape[1]):
        indexes = np.where(dataset[:, 0] == corporates[0, iii])[0]  # dataset corporate
        corporates[1, iii] = indexes[0]
        corporates[2, iii] = indexes[-1]

    corporates = np.int_(corporates) # force them to be int
    return corporates

########################################################################################################################
########################################################################################################################


def confidenceRegionHawkes(dataset, corporates, alphaHat, tauHat, qHat, endDate, minRating, M, alpha):
    """
    function computing the confidence interval for the transitions
    probabilities qij at a level 𝛼=5%
    
    INPUT
    dataset: cleaned DRD Moody's Dataset 
    corporates: matrix having in the first row the list of all the c ∈ C
                              in the second row the first index of c in the dataset
                              in the third row the last index of c in the dataset
    alphaHat:  α parameter from Hawkes calibration
    tauHat: τ parameter from Hawkes calibration
    qHat: transition probability matrix from Hawkes calibration
    enddate: final date of observation
    minRating: minimum rate from where momentum starts 
    M: number of Simulations
    alpha: level of confidence
        % OUTPUT 
    alphaHat_sim : vector of alpha of simulated datasets
    tauHat_sim :vector of tau of simulated datasets
    qHat_sim : q matrix of simulated datasets
    interval_qHwk : confidence interval on qHat_sim
    
    FUNCTION USED
    simulationHawkes
    findCorporates
    calibrationCTM 
    calibrationHawkes

    """

    qHat_sim = np.full((7, 8, M), np.nan)
    alphaHat_sim = np.full((M, 1), np.nan)
    tauHat_sim = np.full((M, 1), np.nan)
    interval_qHwk = np.empty((7, 8, 2))

    for k in range(M):
        dataset_sim = simulationHawkes(dataset, corporates, qHat, minRating, alphaHat, tauHat)
        corporates_simHkw = findCorporates(dataset_sim)
        _, R_Hwk, N_Hwk = calibrationCTM(dataset_sim)
        alphaHat_sim[k], tauHat_sim[k], qHat_sim[:, :, k] = calibrationHawkes(
            dataset_sim, corporates_simHkw, alphaHat, tauHat, endDate, R_Hwk, N_Hwk, minRating
        )

    interval_qHwk = np.quantile(qHat_sim, [alpha/2, 1 - alpha/2], axis=2)
    return alphaHat_sim, tauHat_sim, qHat_sim, interval_qHwk

########################################################################################################################
########################################################################################################################


def plotConfidenceRegion(alphaHat_sim, tauHat_sim, alpha, M):
    '''
    function plotting ellipsoidal and Bonferroni confidence regions of alpha and tau simulated
    '''
    mu = np.array([np.mean(tauHat_sim), np.mean(alphaHat_sim)])
    V = np.cov(tauHat_sim, alphaHat_sim)
    p = len(V)  # number of parameters needed for the Bonferroni correction

    ## Bonferroni CR
    a = t.ppf(1 - alpha / (2 * p), M - 1) * np.sqrt(V[0, 0])  # tau semiaxis
    b = t.ppf(1 - alpha / (2 * p), M - 1) * np.sqrt(V[1, 1])  # alpha semiaxis
    x1 = mu[0] - a
    x2 = mu[0] + a
    y1 = mu[1] - b
    y2 = mu[1] + b
    plt.figure()
    plt.plot(tauHat_sim, alphaHat_sim, 'bo')

    plt.plot([x1, x2], [y1, y1], color='red')
    plt.plot([x2, x2], [y1, y2], color='red')
    plt.plot([x2, x1], [y2, y2], color='red')
    plt.plot([x1, x1], [y2, y1], color='red')
    plt.show()

    ## Ellipsoidal CR

    eigenvalues, eigenvectors = np.linalg.eig(V)
    a = np.sqrt(eigenvalues[0] * chi2.ppf(1 - alpha / 2, 2))
    b = np.sqrt(eigenvalues[1] * chi2.ppf(1 - alpha / 2, 2))
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    plt.figure()
    plt.plot(tauHat_sim, alphaHat_sim, 'bo')
    plot_ellipse(mu, [a, b], theta)

    ## Bonferroni CR
    plt.plot([x1, x2], [y1, y1], color='red')
    plt.plot([x2, x2], [y1, y2], color='red')
    plt.plot([x2, x1], [y2, y2], color='red')
    plt.plot([x1, x1], [y2, y1], color='red')
    plt.xlabel('Tau')
    plt.ylabel('Alpha')
    plt.title('Confidence Region')
    plt.show()

########################################################################################################################
########################################################################################################################


def plot_ellipse(mu, semi_axes_lengths, major_axis_direction):
    '''
    Function that plots the rotated and translated ellipse
    '''
    a = semi_axes_lengths[0]
    b = semi_axes_lengths[1]
    theta = major_axis_direction

    t = np.linspace(0, 2 * np.pi, 100)

    x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
    y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)

    plt.plot(mu[0] + x, mu[1] + y, color='blue')


########################################################################################################################
########################################################################################################################


def confidenceIntervalCTM(dataset, corporates, qmatrix, M, alpha):
    '''
    Function computing the confidence interval for the transitions probabilities qij at a level alpha=5%

    INPUT
    dataset: cleaned DRD Moody's Dataset
    corporates: matrix of corporates details
    qmatrix: matrix of transitions probabilities
    M: number of simulations

    OUTPUT
    qmatrix_final: matrix of the confidence interval for qij, which is 7x8x2
    where in the first 'floor' are reported the quantile 5% and in the other 95%

    FUNCTION USED
    simulationCTM
    calibrationCTM
    '''

    qHat_sim = np.full((7, 8, M), np.nan)
    interval_qHwk = np.empty((7, 8, 2))

    for k in range(M):
        dataset_sim = simulationCTM(dataset, corporates, qmatrix)
        qHat_sim[:, :, k], R, N = calibrationCTM(dataset_sim)

    interval_qHwk = np.quantile(qHat_sim, [alpha/2, 1 - alpha/2], axis=2)

    return interval_qHwk

########################################################################################################################
########################################################################################################################


def normalityCheck(alphaHat_sim, tauHat_sim):
    '''
    function that checks for normality both of the input parameters and the
    joint normality

    INPUT
    alphaHat_sim: array of α parameters from simulations
    tauHat_sim: array of τ parameters from simulations
    '''

    # check for normality of the parameters
    K_alpha = kstest((alphaHat_sim-np.mean(alphaHat_sim))/np.std(alphaHat_sim), 'norm')
    K_tau = kstest((tauHat_sim-np.mean(tauHat_sim))/np.std(tauHat_sim), 'norm')
    print()
    print('Normality α: ', K_alpha)
    print('Normality τ: ', K_tau)

    # QQ plots
    probplot(alphaHat_sim, dist="norm", plot=pylab)
    pylab.show()
    probplot(tauHat_sim, dist="norm", plot=pylab)
    pylab.show()

    # Henze-Zirkler multivariate normality test
    data = np.matrix([alphaHat_sim, tauHat_sim])
    mult_norm = pg.multivariate_normality(data.T, alpha=.05)
    print()
    print(mult_norm)
