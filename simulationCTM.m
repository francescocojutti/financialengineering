function [dataset_sim] = simulationCTM(dataset,corporates,qmatrix)
% function simulating a fake dataset with same corporates, start rating and
% period of observations as the real dataset. It applies the Algorithm
% described in Baviera 4.1
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix of corporates details  
% qmatrix: matric of transition probabilities
% startdate: start date of our analysis
% enddate: end date of our analysis 
%
% OUTPUTS
% dataset_sim: dataset simulated


% initialization of simulated dataset
dataset_sim = NaN.*ones(5*1e4,4);% init
datasetCorporate_sim = NaN.*ones(30,4); % initialization of simulated datastCorporate for c
totcount = 0; % total number of transition simulated, i.e. new lines of the simulated dataset
%ratings = 1:8; % set of ratings
%% loop for replicating all the corporates c ∈ C
for ii = 1:length(corporates)
    c = corporates(1,ii); % corporate c
    datasetCorporate = dataset(corporates(2,ii):corporates(3,ii),:); % take the dataset just for the company c
    tStart = datasetCorporate(1,2); % start date of the simulation
    tEnd = datasetCorporate(end,3); % end date of the simulation  
    ratingStart = datasetCorporate(1,end); % initial rating
    
    % initalization of Simulation Algorithm
    rating_i = ratingStart; % starting rate
    s = tStart; % starting time
    %flag = 0; % reset the flag = 0
    datasetCorporate_sim = NaN.*datasetCorporate_sim; 
    count = 0; % reset of simulated datasetCorporate for c
    
    %% Simulation Algorithm
    while s < tEnd % while jump-times are within the simulation period
        if rating_i == 8 % If the rating is Default
            % add the jump to the dataset
            rowDataset = [c, s, tEnd, rating_i]; % extend D=8 up to tEnd
            count = count+1; % update the counter
            datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
            break % exit the while loop
        
        else % rating_i ~= 8 %and(or(all(isnan(datasetCorporate_sim)), flag==1),
            %flag = 0; % reset the flag
            intensity = sum(qmatrix(rating_i,:)); % intensities λ = Σq, i ≠ j
            dt = -1/intensity .* log(rand); % Δt interarrival time as exponential distribution
            s1 = s + floor(dt*365); % update time s
            if s1 >= tEnd % the jump-time is after tEnd
                % extend i-th rating up to tEnd
                rowDataset = [c, s, tEnd, rating_i];
                count = count+1; % update the counter
                datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
                break % exit the while loop
            else % the jump-time is valid
                u = rand*intensity; % I draw an Uniform r.v. between [0,λ]
                % find the arrival rating
                segment = intensity-cumsum(qmatrix(rating_i,:)); % λ - Σ(k)q
                rating_j = find(u >= segment,1,"first"); % find k s.t. U ≥ λ - Σ(k)q
                % add the jump to the dataset
                rowDataset = [c, s, s1, rating_i];
                count = count+1; % update the counter
                datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
                rating_i = rating_j; % update rating i
                s = s1; % update time s = s1
                continue
            end
         end
    end % while end
    % add the simulated datasetCorporate to the whole dataset
    dataset_sim(totcount+1:totcount+count,:) = datasetCorporate_sim(~isnan(datasetCorporate_sim(:,1)),:);
    totcount = totcount+count; % update the counter
end
dataset_sim = dataset_sim(~isnan(dataset_sim(:,1)),:); % just take the elements that are ≠ NaN