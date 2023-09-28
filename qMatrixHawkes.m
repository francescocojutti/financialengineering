function [qHat] = qMatrixHawkes(Itimes,alpha,tau,R,Rvec,N,minRating)
% Function that computes Q matrix following a Multidimensional Hawkes Model for credit rating migrations
%
%
% INPUTS
% Itimes: 5-dimensional matrix with Δtimes for I for the LogLikelihood computation:
%         dim1: consecutive downgrades
%         dim2: sets of consecutive downgrades
%         dim3: 1st layer for Δt1 and 2nd layer for Δt2
%         dim4: full corporates C set
%         dim5: i start rating (from minRating+1 to 7)
% alpha: calibrated α parameter
% tau: calibrated τ parameter
% R: R vector from CTM calibration
% Rvec: R vector last components dimension-adjusted
% N: N vector from CTM calibration
% minRating: minimum rating from which there is the downgrade momentum effect
%
% 
% OUTPUTS
% qHat: calibrated Q matrix


qHat = N./R; % init with qHat = qmatrix from CTM

% denominator as function (handle):
% Denominator_i = R + αI
Denominator_i = @(alpha,tau,ith) Rvec(1,1,1,1,ith) + alpha.*sum( tau.*sum( exp(-Itimes(:,:,1,:,ith)./tau) - exp(-Itimes(:,:,2,:,ith)./tau) ,[1 2],'omitnan') ,4,'omitnan');

for i = minRating+1:7 % loop on i-th starting ratings
    denominator = Denominator_i(alpha,tau,i-minRating);
    qHat(i,i+1:end) = N(i,i+1:end)./denominator;
end