function [qmatrix_final] = confidenceIntervalCTM(dataset,corporates,qmatrix, M, alpha)
% function computing the confidence interval for the transitions
% probabilities qij at a level alpha=5%
% 
% INPUT
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix of corporates details
% qmatrix: matrix of transitions probabilities 
% M: number of simulations
%
% OUTPUT 
% qmatrix_final: matrix of the confidence interval for qij, which is 7x8x2
% where in the first 'floor' are reported the quantile 5% and in the other 95% 
%
% FUNCTION USED
% simulationCTM
% calibrationCTM

qmatrix_cal = NaN.*ones(7,8,M);
parfor n = 1:M
    [dataset_sim] = simulationCTM(dataset,corporates,qmatrix);
    qmatrix_cal(:,:,n) = calibrationCTM(dataset_sim);
end
qmatrix_final = quantile(qmatrix_cal,[alpha/2 1-alpha/2],3);