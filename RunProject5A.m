%% Project 5

% Group A, AY2022-2023
% Algeri Matteo, Cojutti Francesco, Fouad Davide

clear;
close all;
clc;

%% Conventions
seed = 5; rng(seed);

%% Load data
matrix = load('dataset_num_adj.mat');
startDate = datenum('1-Jan-1983');
endDate = datenum('31-Jan-2020');

%% Original version
flag = 1; % Re-entry after default as a new issuer  
minRating = 3; % minimum rating after which consecutive downgrades originate a downgrading momentum

%% First facultative version
% flag = 2; % default considered as a pure absorbing state, no re-entry after default  
% minRating = 3; % BBB is the minimum rating after which consecutive downgrades originate a downgrading momentum

%% Second facultative version
% flag = 1; % Re-entry after default as a new issuer  
% minRating = 2; % A is the minimum rating after which consecutive downgrades originate a downgrading momentum

%% 1) Data Cleaning 
[dataset] = cleanDataset(matrix.dataset_num_adj(:,[1 2 3 5]),startDate,endDate,flag);

%% 2) Calibration 
% CTM
[qCTM,R,N] = calibrationCTM(dataset);

% Hawkes 
alpha0 = 2; tau0 = 0.5;
[corporates] = findCorporates(dataset);
[alphaHwk, tauHwk, qHwk] = calibrationHawkes(dataset,corporates,alpha0,tau0,endDate,R,N,minRating);

% Plot of the likelihood function of the multi-d Hawkes
M = 50; % we used M = 100 for the plots
[alphas,betas,logL] = plotLogLikeHawkes(dataset,corporates,endDate,R,N,minRating,M);

 %% 3) Simulation
% CTM
[dataset_simCTM] = simulationCTM(dataset,corporates,qCTM);
[q_simCTM,~,~] = calibrationCTM(dataset_simCTM);

% Hawkes
dataset_sim = simulationHawkes(dataset,corporates,qHwk,minRating,alphaHwk,tauHwk);
[corporates_simHkw] = findCorporates(dataset_sim);
[~,R_Hwk,N_Hwk] = calibrationCTM(dataset_sim);
[alpha_simHwk, tau_simHwk, q_simHwk] = calibrationHawkes(dataset_sim,corporates_simHkw,alpha0,tau0,endDate,R_Hwk,N_Hwk,minRating);

%% 4)  confidence region the parameters 𝛼, 𝜏 and confidence intervals for the transition intensities qij
M=1000; alpha = 0.05;
% CTM
interval_qCTM = confidenceIntervalCTM(dataset,corporates,qCTM,M,alpha);

% Hawkes
[alphaHat_sim, tauHat_sim, qHat_sim, interval_qHwk] = confidenceRegionHawkes(dataset,corporates,alphaHwk,tauHwk,qHwk,endDate, minRating,M,alpha);

%%
normalityCheck(alphaHat_sim, tauHat_sim)
plotConfidenceRegion(alphaHat_sim,tauHat_sim,alpha, M);