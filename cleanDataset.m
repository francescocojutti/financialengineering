function [dataset_cleaned] = cleanDataset(dataset, startdate, enddate, flag)
% function executing the following commands for cleaning the dataset:
% -replacing 'NaN' with enddate 
% -selection of the observations between startdate and enddate
% -aggregation of notches in the new rating system
%   1)	1
%   2)	2 3 4
%   3)	5 6 7
%   4)	8 9 10
%   5)	11 12 13
%   6)	14 15 16
%   7)	17 18 19	20 21
%   8)	23
%   WR  24
% -handle the WR as written in Baviera Note
%
% INPUTS
% dataset: original DRD Moody's Dataset 
% startDate: initial date of the analysis  
% endDate: final date of the analysis 
% flag: 1(default not considered as a pure absorbing state)
%       2(default considered as a pure absorbing state)
% 
% 
% OUTPUTS
% dataset_cleaned
% 
% FUNCTION USED 
% mergeRows

%% Replace NaN with enddate 
dataset(isnan(dataset(:,3)),3) = enddate;

%% Select obs inside the considered time interval 
dataset = dataset((dataset(:,2) < enddate) ... % obs initial date anterior enddate
    & (dataset(:,3) > startdate),:);         % obs final date posterior startdate (decidere se considerare anche gli uguali) 35 vs 23
% Replace obs initial date w startdate when initial date anterior startdate 
dataset(dataset(:,2) < startdate, 2) = startdate; 
% Replace obs final date w enddate when final date posterior enddate 
dataset(dataset(:,3) > enddate, 3) = enddate;

%% Replace ratings w new rating system 
% Replace obs with rating==22(WR) w rating 24
dataset((dataset(:,end) == 22),end) = 24;
% Aggregiation of notches in new 7+1(D) classes
idxR = [0 1 4 7 10 13 16 21 23]; % indexes of ratings intervals
for i = 1:length(idxR)-1
    idx = (dataset(:,end) > idxR(i)) & (dataset(:,end) <= idxR(i+1)); % indexes obs inside the i-th interval
    dataset(idx,end) = i;                                             % replace w i-th rating
end
clear("idx")

%% Handle D and WR wrt different situation

if flag == 1 % D is not an absorbing state
    Default = (dataset(:,end)==8); % D
    dataset(:,1) = dataset(:,1)+cumsum([false;Default(1:end-1)]); % Reentry after default as an independent issuer 
    dataset(Default,3) = enddate; % extend final date of default to enddate 
elseif flag == 2 % D is an absorbing state
    corporates = unique(dataset(:,1)); % vector of idnumber of each corporate reported one time and sorted
    dataset_new = [];
    for i = 1:length(corporates)
        c = corporates(i);
        datasetCorporate = dataset(dataset(:,1)==c,:); % take the dataset just for the company c
        if sum(datasetCorporate(:,end)==8) == 0
            dataset_new = [dataset_new ; datasetCorporate];
        else
            firstDefault = find(datasetCorporate(:,end)==8,1,'first'); % index of the first default of a corporate 
            datasetCorporate(firstDefault,3) = enddate;
            dataset_new = [dataset_new ; datasetCorporate(1:firstDefault,:)];
        end
    end
    dataset = dataset_new;
end

% WR as first observation of the corporate
firstWR = (dataset(2:end,end)==24 ...           % WR
    & (dataset(1:end-1,1)~=dataset(2:end,1)) ); % first obs of the issuer, i.e. different issuer number
dataset([false; firstWR(1:end-1)],:) = [];      % 

% WR as last observation of the corporate
lastWR = (dataset(1:end-1,end)==24 ...          % WR
    & (dataset(1:end-1,1)~=dataset(2:end,1)) ); % last obs of the issuer, i.e. different issuer number
dataset(lastWR,:) = [];

%% Condition 1 Baviera
% Transitions to be adjusted: those who default from WR less than nYears (1 year) after their rating was withdrawn
nDay = 365;
cond1 = (dataset(1:end-1,end)==24) ...                 % from WR
    & (dataset(2:end,end)==8) ...                      % to D
    & (dataset(1:end-1,1)==dataset(2:end,1)) ...       % same issuer number
    & (dataset(1:end-1,3)-dataset(1:end-1,2) <= nDay);  % WR last less than nDay
% adjust dates
dataset([false; cond1(1:end-1)], 2) = dataset(cond1, 2);
% remove WR rows
dataset(cond1, :) = [];

%% Condition 2 Baviera
% Transitions in D from WR with timelag greater than nDays(365)
cond2 = (dataset(1:end-1,end)==24) ...                 % from WR
    & (dataset(2:end,end)==8) ...                      % to D
    & (dataset(1:end-1,1)==dataset(2:end,1)) ...       % same issuer number
    & (dataset(1:end-1,3)-dataset(1:end-1,2) > nDay); % WR last more than nDay
idx = [find(cond2)+1;find(cond2)];
dataset(idx , :) = [];                                  % cancel WR & D
clear("idx")

% D as first obs of the corporate
firstD = (dataset(2:end,end) == 8)...           % D
    & (dataset(1:end-1,1)~=dataset(2:end,1)); % first obs of the issuer, i.e. different issuer number
idx = find(firstD)+1;
dataset(idx,:) = [];% cancel defaults
clear("idx")

%% Condition 3 Baviera
% Cancel WR that last <nDay and is followed by a new rating
cond3 = ( (dataset(2:end-1,end)==24) ...                    % WR
    & (dataset(2:end-1,3)-dataset(2:end-1,2) <= nDay) ...   % WR last less than nDay
    & (dataset(2:end-1,1)==dataset(3:end,1)) ...            % same following issuer 
    & (dataset(1:end-2,1)==dataset(2:end-1,1)));            % same previous issuer
idx = find(cond3);
dataset(idx,3) = dataset(idx+2,2);
dataset(idx+1,:) = []; 
clear("idx")

%% Condition 4 Baviera
% Cancel WR that last more than nDay and new issuer from following rating
cond4 = (dataset(:,end)==24 ...             % WR
    & (dataset(:,3)-dataset(:,2) > nDay)); % WR last more than nDay
dataset(:,1) = dataset(:,1)+cumsum([false;cond4(1:end-1)]);
dataset(cond4, :) = [];

%% Merge consecutive obs w same rating
corporates = unique(dataset(:,1));% vector of idnumber of each corporate reported one time and sorted
dataset_cleaned = [];
for i = 1:length(corporates)
    c = corporates(i);
    datasetCorporate = dataset(dataset(:,1)==c,:); % take the dataset just for the company c
    [dataset_cleaned] = mergeRows(datasetCorporate,dataset_cleaned);
end