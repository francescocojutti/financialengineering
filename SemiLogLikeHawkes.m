function [Itimes,Jtimes] = SemiLogLikeHawkes(dataset,corporates,minRating,T,dim1,dim2)
% Function that conmputes the Δtimes for I and J for the LogLikelihood computation
%
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix having in the first row the list of all the c ∈ C
%                           in the second row the first index of c in the dataset
%                           in the third row the last index of c in the dataset
% minRating: minimum rating from which there is the downgrade momentum effect
% T: end Date of analysis
% dim1: size of dimension 1 of I and J matrices
% dim2: size of dimension 2 of I and J matrices
%
% 
% OUTPUTS
% Itimes: 5-dimensional matrix with Δtimes for I for the LogLikelihood computation
% Jtimes: 5-dimensional matrix with Δtimes for J for the LogLikelihood computation
%     The matrices dimensions are as follows:
%     Itimes: dim1: consecutive downgrades
%             dim2: sets of consecutive downgrades
%             dim3: 1st layer for Δt1 and 2nd layer for Δt2
%             dim4: full corporates C set
%             dim5: i start rating (from minRating+1 to 7)
%     Jtimes: dim1: consecutive downgrades
%             dim2: sets of consecutive downgrades
%             dim3: j arrival ratings
%             dim4: full corporates C set
%             dim5: i start rating (from minRating+1 to 7)
%
%
% FUNCTIONS USED
% helpParJ
% computeIic


%% preallocating memory for matrices (preallocating for speed)
lenCo = length(corporates); % number of corporates
Itimes = NaN.*ones(dim1,dim2,2,lenCo,7-minRating); % matrix of times for I
Jtimes = NaN.*ones(dim1,dim2,8-minRating-1,lenCo,7-minRating); % matrix of times for J
TTI = NaN.*ones(dim1,dim2,2); % auxiliary matrix for I computation (first 3 dimensions)
TTJ = NaN.*ones(dim1,dim2); % auxiliary matrix for J computation (first 2 dimensions)
IvecC = NaN.*ones(dim1,dim2,2,lenCo); % auxiliary matrix for corporates data for I computation (+4th dimension)
JmatC = NaN.*ones(dim1,dim2,8-minRating-1,lenCo); % auxiliary matrix for corporates and j-th rating sum data for j computation
J3d = NaN.*ones(dim1,dim2,8-minRating-1); % auxiliary first 3 dimensions of J vector of times
%% loop on i-th starting ratings for LogLikelihood computation

for i = minRating+2:7 % loop on i-th starting ratings
    % resetting matrices
    IvecC = NaN.*IvecC;
    JmatC = NaN.*JmatC;
    for iii = 1:lenCo % loop on corporates c ∈ C
        datasetCorporate = dataset(corporates(2,iii):corporates(3,iii),:); % select dataset relative to the corporate(iii)
        JmatC(:,:,:,iii) = helpParJ(i,datasetCorporate,minRating,J3d,TTJ); % store data for J for each corporate
        IvecC(:,:,:,iii) = computeIic(i,datasetCorporate,minRating,T,TTI); % store data for I for each corporate
    end
    Jtimes(:,:,:,:,i-minRating) = JmatC; % add the data to the matrix
    Itimes(:,:,:,:,i-minRating) = IvecC; % add the data to the matrix

end

%% Matrix dimension reduction
% In order to improve fmincon/fminsearch speed we reduce the matrices
% dimensions, so that basic operations are much faster

% initialization
ItimesNew = NaN.*Itimes;
JtimesNew = NaN.*Jtimes;
lenI = zeros(7-minRating-1,1);
lenJ = zeros(7-minRating-1,1);

for d5 = 1:7-minRating % for loop on 5-th dimension
    % find submatrixes of dt where there isn't any NaN (find corporates with non NaN data)
    idxI = any(~isnan(Itimes(:,:,:,:,d5)),[1 2 3]);
    idxJ = any(~isnan(Jtimes(:,:,:,:,d5)),[1 2 3]);
    % reshape vector idx
    auxI = reshape(idxI,numel(idxI),1);
    auxJ = reshape(idxJ,numel(idxJ),1);
    % store the clean data in a temp matrix
    tmpI = Itimes(:,:,:,auxI,d5);
    tmpJ = Jtimes(:,:,:,auxJ,d5);
    % store the number of corporates with useful data (different from all NaNs)
    lenI(d5) = length(tmpI(1,1,1,:));
    lenJ(d5) = length(tmpJ(1,1,1,:));
    % finally store the clean data in the new matrices
    ItimesNew(:,:,:,1:lenI(d5),d5) = tmpI;
    JtimesNew(:,:,:,1:lenJ(d5),d5) = tmpJ;
end
% cut the final matrixes to the max number of useful corporates
Itimes = ItimesNew(:,:,:,1:max(lenI),:);
Jtimes = JtimesNew(:,:,:,1:max(lenJ),:);
