function [J] = helpParJ(i,datasetCorporate,minRating,J3d,TT)
% The function returns the Δt matrix for the i-th starting rating, for
% every j>i arrival rating for input corporate c
%
%
% INPUTS
% i: initial rating at jump-time
% datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
% minRating: minimum rating from which there is the downgrade momentum effect
% J3d: J matrix of 3 dimensions
% TT: auxiliary preallocated 2-dimensional matrix
%
%
% OUTPUT
% J: Δt matrix for J up to first 3 dimensions for the corporate c
%
%
% FUNCTION USED
% computeJijc


J = NaN.*J3d; % init

for j = i+1:8 % loop on j>i arrival ratings
    J(:,:,j-i) = computeJijc(i,j,datasetCorporate,minRating,TT); % store data for the c corporate for every j-th arrival rating
end