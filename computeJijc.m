function [TT] = computeJijc(i,j,datasetCorporate,minRating,TT)
% Function that computes the first 2 dimensions of Jtimes for jumps from i
% to j for corporate c
%
%
% INPUTS
% i: initial rating at jump-time
% J: arrival rating at jump-time
% datasetCorporate: cleaned DRD Moody's Dataset just for the company c ∈ C
% minRating: minimum rating from which there is the downgrade momentum effect
% TT: auxiliary preallocated 2-dimensional matrix
%
%
% OUTPUT
% TT: filled matrix with all the sets of consecutive downgrades of interest


TT = NaN.*TT; % init

if any((datasetCorporate(:,end) == i) == 1) % if there is at least one i-th rating
    % indexes of i-th rating for the corporate c arriving in j
    indexesi = find(and(datasetCorporate(1:end-1,end) == i, datasetCorporate(2:end,end) == j));
    indexesi = indexesi(indexesi>1);
    for i = 1:length(indexesi) % loop on the sets of consecutive downgrades
       minR = datasetCorporate(1:indexesi(i),end) > minRating; % boolean: 1: ratings > minrating
       is_downgrade = minR(1:end-1).*(diff(datasetCorporate(1:indexesi(i),end)) > 0); % boolean: 1: it is downgrade of interest
       idxtmp = [find(is_downgrade == 0,1,'last'), 0]; % find index before starting index
       idxstart = idxtmp(1)+1; % starting index
       if idxstart < indexesi(i) % if the index is acceptable
          dt = (datasetCorporate(indexesi(i),3)-datasetCorporate(idxstart:indexesi(i)-1,3))/365; % Δt computations
          TT(1:length(dt),i) = dt; % adding Δt to the matrix
       end
    end
end
