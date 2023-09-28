function [] = normalityCheck(alphaHat_sim, tauHat_sim)
% function that checks for normality both of the input parameters and the
% joint normality
% 
% INPUT
% alphaVec: array of α parameters from simulations
% tauVec: array of τ parameters from simulations
%

% check for normality of the parameters
[h,p] = kstest((alphaHat_sim-sum(alphaHat_sim)/numel(alphaHat_sim))./std(alphaHat_sim));
if h == 0
    fprintf('\nAlpha is normally distributed, with p-value: %f\n', p)
end
[h,p] = kstest((tauHat_sim-sum(tauHat_sim)/numel(tauHat_sim))./std(tauHat_sim));
if h == 0
    fprintf('\nTau is normally distributed, with p-value: %f\n', p)
end

% QQ plots
figure()
qqplotPretty(alphaHat_sim)
figure()
qqplotPretty(tauHat_sim)

% check for joint normality
HZmvntest([alphaHat_sim, tauHat_sim],cov([alphaHat_sim, tauHat_sim]))