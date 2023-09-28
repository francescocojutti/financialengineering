function [dataset_out] = mergeRows(datasetCorporate,dataset_merged)
% function that aggregates same cosnecutive rating after the change of the
% rating evaluation
%
% INPUT 
% datasetCorporate:dataset of the corporate c
% dataset_merged: dataset where consecutive rows with same rating 
% 
% OUTPUT
% dataset_out: dataset merged

tmpMat = datasetCorporate(1,:); % initialization with first row of the corporate
for ii = 2:length(datasetCorporate(:,1))
    if datasetCorporate(ii,end) == datasetCorporate(ii-1,end) %consecutive rating
        tmpMat(end,3) = datasetCorporate(ii,3); % enddate of last rating transition is extended
    else
        tmpMat(end+1,:) = datasetCorporate(ii,:); % add the rating transition
    end
end
dataset_out = [dataset_merged; tmpMat];