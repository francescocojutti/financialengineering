function [corporates] = findCorporates(dataset)
% function finding all the corporates in the dataset and returning a matrix
% 3xNcorporates where:
% corporates(1,:) = corporate names (indexed numbers)
% corporates(2,:) = first indexes of dataset for each company
% corporates(3,:) = last indexes of dataset for each company

tmp = 1:length(dataset(:,1));
corporates(1,:) = unique(dataset(:,1)); % find all corporates
corporates = [corporates;zeros(2,length(corporates))];
for iii = 1:length(corporates)
    indexes = tmp(dataset(:,1)==corporates(1,iii)); % dataset corporate

    corporates(2,iii) = indexes(1);
    corporates(3,iii) = indexes(end);
end
