function [alphaHat, tauHat, qHat] = calibrationHawkes(dataset,corporates,alpha0,tau0,T,R,N,minRating)
% Function that calibrates a Multidimensional Hawkes Model for credit rating migrations
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix having in the first row the list of all the c ∈ C
%                           in the second row the first index of c in the dataset
%                           in the third row the last index of c in the dataset
% alpha0: starting guess for α parameter calibration
% tau0: starting guess for τ parameter calibration
% T: end Date of analysis
% R: R vector from CTM calibration
% N: N vector from CTM calibration
% minRating: minimum rating from which there is the downgrade momentum effect
%
% 
% OUTPUTS
% alphaHat: calibrated α parameter
% tauHat: calibrated τ parameter
% qHat: calibrated Q matrix
%
%
% FUNCTIONS USED
% SemiLogLikeHawkes
% qMatrixHawkes


% init
dim1 = length(minRating+1:7); % dimension 1 matrix, max length of consecutive downgrades
dim2 = floor(max(corporates(3,:)-corporates(2,:))/2)+1; % dimension 2 matrix

% SemiLoghLikelihood computation (only the part that depends from α and τ)
% Itimes, Jtimes: matrices of Δtimes to be plugged at the exponentials in the formula
[Itimes,Jtimes] = SemiLogLikeHawkes(dataset,corporates,minRating,T,dim1,dim2);

% Dimensions adjustment of N and R for future computation 
Nvec = zeros(1,1,1,1,7-minRating);
for i = minRating+1:7
    Nvec(1,1,1,1,i-minRating) = sum(N(i,i+1:end));
end
Rvec(1,1,1,1,:) = R(minRating+1:7);

% Funcions (handle) for first (I) and second (J) components of the SemiLoghLikelihood
Icomp = @(alpha,tau) sum( -Nvec.*log(Rvec + alpha.*sum( tau.*sum( exp(-Itimes(:,:,1,:,:)./tau) - exp(-Itimes(:,:,2,:,:)./tau) ,[1 2],'omitnan') ,4,'omitnan')) );
Jcomp = @(alpha,tau) sum( log(1+alpha.*sum( exp(-Jtimes./tau) ,1,'omitnan')) ,'all','omitnan');

% function to minimize for finding the optimal parameters
LogLiketoMax = @(param) -(Jcomp(param(1),param(2)) + Icomp(param(1),param(2)));

param0 = [alpha0;tau0]; % initial guess
options = optimset('Display','off'); % fminsearch options
% Optimal parametrs finding
[param_opt] = fminsearch(LogLiketoMax,param0,options);

alphaHat = param_opt(1); 
tauHat = param_opt(2);
% Q matrix computation
[qHat] = qMatrixHawkes(Itimes,alphaHat,tauHat,R,Rvec,N,minRating);