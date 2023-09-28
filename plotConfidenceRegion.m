function [] = plotConfidenceRegion(alphaHat_sim, tauHat_sim, alpha, M)
% function plotting ellipsoidal and Bonferroni confidence regions of alpha and tau simulated

mu = [mean(tauHat_sim);mean(alphaHat_sim)];
V= cov(tauHat_sim, alphaHat_sim);
p=length(V);
%% Bonferroni CR
figure()
hold on
a = icdf('t', 1-alpha/(2*p), M-1) *sqrt(V(1,1)); %tau semiaxis
b = icdf('t', 1-alpha/(2*p), M-1) *sqrt(V(2,2)); %alpha semiaxis
x1= mu(1)-a;
x2 =mu(1)+a;
y1= mu(2)-b;
y2= mu(2)+b;
line([x1,x2], [y1, y1], 'Color', 'red')
line([x2,x2], [y1,y2], 'Color', 'red')
line([x2,x1], [y2, y2], 'Color', 'red')
line([x1,x1], [y2, y1], 'Color', 'red')
plot(tauHat_sim, alphaHat_sim, 'bo')
xlabel('$\tau$','interpreter','latex','FontSize',20)
ylabel('$\alpha$','interpreter','latex','FontSize',20)
title('Bonferroni Confidence Region','interpreter','latex','FontSize',24);

%% Ellipsoidal CR
[vec,lambda] = eig(V);
a= sqrt(lambda(1,1)*chi2inv(1-alpha/2,2));
b= sqrt(lambda(2,2)*chi2inv(1-alpha/2,2));
theta = atan(vec(1,2)/vec(1,1));  % Direction of the major semi-axis
figure()
hold on
plot(tauHat_sim, alphaHat_sim, 'bo')
plot_ellipse(mu, [a b], theta);

%% Bonferroni CR
a = icdf('t', 1-alpha/(2*p), M-1) *sqrt(V(1,1)); %tau semiaxis
b = icdf('t', 1-alpha/(2*p), M-1) *sqrt(V(2,2)); %alpha semiaxis
x1= mu(1)-a;
x2 =mu(1)+a;
y1= mu(2)-b;
y2= mu(2)+b;
line([x1,x2], [y1, y1], 'Color', 'red')
line([x2,x2], [y1,y2], 'Color', 'red')
line([x2,x1], [y2, y2], 'Color', 'red')
line([x1,x1], [y2, y1], 'Color', 'red')
% Add labels and title to the plot
xlabel('$\tau$','interpreter','latex','FontSize',20)
ylabel('$\alpha$','interpreter','latex','FontSize',20)
title('Confidence Region','interpreter','latex','FontSize',24);
legend('Data', 'Elliptical','Bonferroni','Location','northeast','interpreter','latex', 'FontSize', 16)

