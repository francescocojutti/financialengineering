function [alphas,betas,logL] = plotLogLikeHawkes(dataset,corporates,T,R,N,minRating,M)
% Function that plots the LogLikelihood of a Multidimensional Hawkes Model for credit rating migrations
%
%
% INPUTS
% dataset: cleaned DRD Moody's Dataset 
% corporates: matrix having in the first row the list of all the c ∈ C
%                           in the second row the first index of c in the dataset
%                           in the third row the last index of c in the dataset
% T: end Date of analysis
% R: R vector from CTM calibration
% N: N vector from CTM calibration
% minRating: minimum rating from which there is the downgrade momentum effect
% M: number of points in each plot's axis
%
% 
% OUTPUTS
% alphas: α points in the plot
% betas: β points in the plot
% logL: LogLikelihood grid
%
%
% FUNCTIONS USED
% SemiLogLikeHawkes


% init
dim1 = 4; % dimension 1 matrix
dim2 = floor(max(corporates(3,:)-corporates(2,:))/2)+1; % dimension 2 matrix

% SemiLoghLikelihood computation (only the part that depends from α and τ)
% Itimes, Jtimes: matrices of Δtimes to be plugged at the exponentials in the formula
[Itimes,Jtimes] = SemiLogLikeHawkes(dataset,corporates,minRating,T,dim1,dim2);

% Dimensions adjustment of N and R for future computation 
Nvec = zeros(1,1,1,1,7-minRating);
for i = minRating+1:7
    Nvec(1,1,1,1,i-minRating) = sum(N(i,i+1:end));
end
Rvec(1,1,1,1,:) = R(minRating+1:7);

% Funcions (handle) for first (I) and second (J) components of the SemiLoghLikelihood
Icomp = @(alpha,tau) sum( -Nvec.*log(Rvec + alpha.*sum( tau.*sum( exp(-Itimes(:,:,1,:,:)./tau) - exp(-Itimes(:,:,2,:,:)./tau) ,[1 2],'omitnan') ,4,'omitnan')) );
Jcomp = @(alpha,tau) sum( log(1+alpha.*sum( exp(-Jtimes./tau) ,1,'omitnan')) ,'all','omitnan');

% function to minimize for finding the optimal parameters
LogLike = @(param) Jcomp(param(1),param(2)) + Icomp(param(1),param(2));

% auxiliary computations
tempN=N;
for i=1:7
    for j=1:8
        if(tempN(i,j)==0)
            tempN(i,j)=1;
        end
    end
end

% init
alphas= linspace(3,6,M);
taus=linspace(0.35,0.95,M);
% alphas= linspace(0.5,3.5,M);
% taus=linspace(0.25,0.75,M);
% alphas = linspace(3,7,M);
% taus = linspace(0.05,0.9,M);
betas= 1./taus;
a=N.*log(R);

% LogLikelihood part 1 (independent from α and β)
log1=-size(dataset,1) + sum(sum(N.*log(tempN))) - sum(sum(a(1:minRating, 1:minRating)));

% LogLikelihood part 2 (function of α and β)
log2=zeros(M,M);
parfor j=1:M
    for k=1:M
         log2(j,k)= LogLike([alphas(k); 1/betas(j)]);
    end
end
logL=log1+log2;

%% Plot surface
figure()
surf(alphas,betas,logL)
title('LogLikelihood $L$($\alpha$,$\beta$)','interpreter','latex','FontSize',24)
xlabel('$\alpha$','interpreter','latex','FontSize',20)
ylabel('$\beta$','interpreter','latex','FontSize',20)
zlabel('LogLikelihood','interpreter','latex','FontSize',20)
