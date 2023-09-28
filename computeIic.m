function [TT] = computeIic(i,datasetCorporate,minRating,T,TT)
% Function that computes the first 3 dimensions of Itimes for jumps from i
% for corporate c
%
%
% INPUTS
% i: initial rating at jump-time
% datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
% minRating: minimum rating from which there is the downgrade momentum effect
% T: end date
% TT: auxiliary preallocated 2-dimensional matrix
%
%
% OUTPUT
% TT: filled matrix with all the sets of consecutive downgrades of interest
%     both for 1st and 2nd layer


TT = NaN.*TT; % init

indexesi = find(datasetCorporate(1:end,end) == i); % indexes of i-th rating for the corporate c
indexesi = indexesi(indexesi>1);
for i = 1:length(indexesi) % loop on the sets of consecutive downgrades
    minR = datasetCorporate(1:indexesi(i),end) > minRating; % boolean: 1: ratings > minrating
    is_downgrade = minR(1:end-1).*(diff(datasetCorporate(1:indexesi(i),end)) > 0); % boolean: 1: it is downgrade of interest
    idxtmp = [find(is_downgrade == 0,1,'last'), 0]; % find index before starting index
    idxstart = idxtmp(1)+1; % starting index
    if idxstart < indexesi(i) % if the index is acceptable
        dt1 = (datasetCorporate(indexesi(i),2)-datasetCorporate(idxstart+1:indexesi(i),2))/365; % Δt1 computations
        dt2 = (min(datasetCorporate(indexesi(i),3),T)-datasetCorporate(idxstart+1:indexesi(i),2))/365; % Δt2 computations
        TT(1:length(dt1),i,1) = dt1; % adding Δt1 to the matrix' first layer
        TT(1:length(dt2),i,2) = dt2; % adding Δt2 to the matrix' second layer
    end
end
