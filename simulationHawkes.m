function [dataset_sim] = simulationHawkes(dataset,corporates,qmatrix,minRating,alpha,tau)
% Function that simulates a new dataset following a Multidimensional Hawkes Process
%
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix having in the first row the list of all the c ∈ C
%                           in the second row the first index of c in the dataset
%                           in the third row the last index of c in the dataset
% qmatrix: Q matrix from Hawkes calibration
% minRating: minimum rating from which there is the downgrade momentum effect
% alpha: α parameter from Hawkes calibration
% tau: τ parameter from Hawkes calibration
% 
% 
% OUTPUTS
% dataset_sim: simulated dataset according to a Multidimensional Hawkes Process


% initialization of simulated dataset
dataset_sim = NaN.*ones(5*1e4,4);% init
datasetCorporate_sim = NaN.*ones(30,4); % initialization of simulated datastCorporate for c
totcount = 0;
options = optimset('Display','off'); % options for fzero
ratings = 1:8; % set of ratings
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
    flag = 0; % reset the flag = 0
    datasetCorporate_sim = NaN.*datasetCorporate_sim; count = 0; % reset of simulated datastCorporate for c
    
    %% Simulation Algorithm
    while s < tEnd % while jump-times are within the simulation period
        
        %% 1) If it is the first iteration or it is NOT a downgrade of interest AND NOT default
        if and(or(all(isnan(datasetCorporate_sim)), flag==1), rating_i ~= 8)
            flag = 0; % reset the flag
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
        
        %% 2) Check for Default
        elseif rating_i == 8 % If the rating is Default
            % add the jump to the dataset
            rowDataset = [c, s, tEnd, rating_i]; % extend D=8 up to tEnd
            count = count+1; % update the counter
            datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
            break % exit the while loop
        
        %% 3) From the second iteration on, check if the last time-jump was a downgrade
        % if it is not the first iteration AND it is an "interesting" downgrade
        elseif and( (rating_i > datasetCorporate_sim(count,end)), (datasetCorporate_sim(count,end) > minRating) )     
            % 1) Compute rating downgrade momentum
            % find consecutive downgrades of interest:
            ratingsCorporate = [datasetCorporate_sim(~isnan(datasetCorporate_sim(:,end)),end);rating_i]; % select ratings of the corporate up to s
            is_downgrade = (diff(ratingsCorporate)>0) .* (ratingsCorporate(1:end-1)>minRating); % find downgrades of interest
            idxtmp = [find(is_downgrade == 0,1,'last'), 0]; % find first index before the downgrade cascade
            idxstart = idxtmp(1)+1; % starting index of the downgrade cascade
            % φ function for λijc
            tj = datasetCorporate_sim(idxstart:count,3);
            ti = s;
            phi = @(t) sum(exp(-(t-tj)/(tau*365))); % φ function Σexp(-Δt/τ) rating momentum
              
            % A) Generate candidate jump with exponential distribution
            u = rand;
            % fun: log(u) + ∫(q + Iqα(φ(x)))
            fun = @(x) log(u) + sum(qmatrix(rating_i,:)*(x-ti)/365 + (ratings>rating_i) .* qmatrix(rating_i,:).*alpha.*tau.*(-phi(x)+phi(ti)),'all');
            % eqn: log(u) + ∫(q + Iqα(φ(x))) == 0 in order to find jump-time x
            x = fzero(@(x) fun(x), 750000,options); % update time x
            s1 = floor(x); % update time s1
            if s1 >= tEnd % the jump-time is after tEnd
                % extend i-th rating up to tEnd
                rowDataset = [c, s, tEnd, rating_i];
                count = count+1; % update the counter
                datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
                break % exit the while loop
            else
                % B) draw an Uniform r.v. between [0,λ]
                intensities_ij = qmatrix(rating_i,:) .* (1+ (ratings>rating_i).*alpha.*phi(s1)); % λ = q + Iqα*Σexp(-Δt/τ)
                intensity = sum(intensities_ij); % intensities λ = Σλ, i ≠ j
                u = rand*intensity; % I draw an Uniform r.v. between [0,λ]
                % find the arrival rating
                segment = intensity-cumsum(intensities_ij); % λ - Σ(k)λ
                rating_j = find(u >= segment,1,"first"); % find k s.t. U ≥ λ - Σ(k)λ
                % add the jump to the dataset
                rowDataset = [c, s, s1, rating_i];
                count = count+1; % update the counter
                datasetCorporate_sim(count,:) = rowDataset; % adding the row to the dataset
                rating_i = rating_j; % update rating i
                s = s1; % update time s = s1
            end
        else % If it is NOT a downgrade of interest, do as in step 1)
            flag = 1;
        end
    end % while end
    % add the simulated datasetCorporate to the whole dataset
    dataset_sim(totcount+1:totcount+count,:) = datasetCorporate_sim(~isnan(datasetCorporate_sim(:,1)),:);
    totcount = totcount+count; % update the counter
end
dataset_sim = dataset_sim(~isnan(dataset_sim(:,1)),:); % just take the elements that are ≠ NaN
