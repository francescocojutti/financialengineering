function [alphaHat_sim, tauHat_sim, qHat_sim, interval_qHwk] = confidenceRegionHawkes(dataset,corporates,alphaHat,tauHat,qHat,enddate,minRating,M,alpha)
% function computing the confidence interval for the transitions
% probabilities qij at a level 𝛼=5%
% 
% INPUT
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix having in the first row the list of all the c ∈ C
%                           in the second row the first index of c in the dataset
%                           in the third row the last index of c in the dataset
% alphaHat:  α parameter from Hawkes calibration
% tauHat: τ parameter from Hawkes calibration
% qHat: transition probability matrix from Hawkes calibration
% enddate: final date of observation
% minRating: minimum rate from where momentum starts 
% M: number of Simulations
% alpha: level of confidence
%
% OUTPUT 
% alphaHat_sim : vector of alpha of simulated datasets
% tauHat_sim :vector of tau of simulated datasets
% qHat_sim : q matrix of simulated datasets
% interval_qHwk : confidence interval on qHat_sim
%
% FUNCTION USED
% simulationHawkes
% findCorporates
% calibrationCTM 
% calibrationHawkes


qHat_sim = NaN*ones(7,8,M);
alphaHat_sim=NaN*ones(M,1);
tauHat_sim=NaN*ones(M,1);
parfor k = 1:M
    dataset_sim = simulationHawkes(dataset,corporates,qHat,minRating,alphaHat,tauHat);
    [corporates_simHkw] = findCorporates(dataset_sim);
    [~,R_Hwk,N_Hwk] = calibrationCTM(dataset_sim);
    [alphaHat_sim(k), tauHat_sim(k), qHat_sim(:,:,k)] = calibrationHawkes(dataset_sim,corporates_simHkw,alphaHat,tauHat,enddate,R_Hwk,N_Hwk,minRating);
end

interval_qHwk = quantile(qHat_sim,[alpha/2 1-alpha/2],3);
