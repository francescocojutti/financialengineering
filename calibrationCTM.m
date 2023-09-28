function [qmatrix,R,N] = calibrationCTM(dataset)
% function computing the estimated transition probabilities for a Continuous Time Markov Model 
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
%
% OUTPUTS
% qmatrix: transition probability matrix
% R: total time up to T that all corporates spend with a rating i
% N: total number of transitions (ij) up to T by all corporates

deltaTvec = (dataset(:,3)-dataset(:,2))/365; % vector of time spent in each rating i 
ratings = 1:8; 
R = ones(7,1);
N = zeros(7,8); 
arrivalRating = [dataset(2:end,end);0]; % final ratings
arrivalRating(dataset(2:end,1) - dataset(1:end-1,1) ~= 0) = 0; % set to 0 the last transition of the companies
for i = 1:length(ratings)-1
    idx = find(dataset(:,end)==i); % find the rating i indexes
    R(i) = sum(deltaTvec(idx)); % sum of total time spent in r rating
    N(i,ratings) = sum(arrivalRating(idx)==ratings); % sum of arrivals in rating j
end
qmatrix = N./R;
