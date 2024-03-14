%% Numerical simulation
% Mathematical model to acheive sustainable forest management 
% Date: 13-03-2024
%%
clc
clear all

% Define parameter values
r = 0.1;    % Intrinsic growth rate
K = 100;    % Carrying capacity
Pi =0.01 ;   % Proportionality constant for population growth due to forest resources
alpha =0.008 ;    % Depletion rate coefficient of forest resources due to population
lambda = 0.1;   % Growth rate coefficient of population pressure
lambda0 = 0.8 ;  % Natural depletion rate coefficient of population pressure
lambda1 =0.1 ;  % Depletion rate coefficient of population pressure due to economic efforts
psi = 0.2 ;  % Growth rate coefficient of economic efforts
psi0 = 0.1 ; % Depletion rate coefficient of economic efforts
phi =0.01 ;  % Implementation rate coefficient of technological efforts
phi0 =0.03 ; % Depletion rate coefficient of technological efforts
s =0.6 ;    % Forest resources growth rate
L =50 ;    % Carrying capacity of forest resources
lambda2 =0.0011 ;  % Population pressure growth rate coefficient due to forest resources
phi1 =0.02 ; % Implementation rate coefficient of technological efforts on forest resources
phi2 =0.0006 ; % Implementation rate coefficient of technological efforts on forest resources squared

% Define initial conditions
B0 = 7.9360;   % Initial density of forest resources
N0 = 100.6348;   % Initial human population density
P0 = 5.3700;   % Initial population pressure
E0 =10.7400 ;   % Initial economic efforts
T0 = 14.0213;   % Initial technological efforts

% Define the time span upto 150 years
t_start=1;
t_end=150;
tspan = [t_start, t_end];

% Define the system of differential equations
ode = @(t, y) [s*y(1)*(1-y(1)/L) - alpha*y(1)*y(2) - lambda2*y(1)^2*y(3) + phi1*y(1)*y(5) + phi2*y(1)^2*y(5);
               r*y(2)*(1-y(2)/K) + pi*alpha*y(1)*y(2);
               lambda*y(2) - lambda0*y(3) - lambda1*y(3)*y(4);
               psi*y(3) - psi0*y(4);
               phi*(L-y(1)) - phi0*y(5)];

% ####### CHANGE ALPHA VALUES HERE ###################
alpha =0.008 ;  
% Solve the differential equations
[t1, Y1] = ode45(ode, tspan, [B0, N0, P0, E0, T0]); % alp=0
[t2, Y2] = ode45(ode, tspan, [B0, N0, P0, E0, T0]); % alp=0.002
[t3, Y3] = ode45(ode, tspan, [B0, N0, P0, E0, T0]); % alp=0.005
[t4, Y4] = ode45(ode, tspan, [B0, N0, P0, E0, T0]); % alp=0.008

% Plot the results (Varying alpha)
figure;
plot(t1, Y1(:,1),'r','LineWidth', 1.5); hold on;
plot(t2, Y2(:,1),'b','LineWidth', 1.5); hold on;
plot(t3, Y3(:,1),'g','LineWidth', 1.5 ); hold on;
plot(t4, Y4(:,1),'k', 'LineWidth', 1.5); hold on;
xlabel('Time');
ylabel('Density/Value');
legend('\alpha=0', '\alpha=0.002', '\alpha=0.005', '\alpha=0.008');

%%

% Define the parameter ranges and initial conditions
phi2_range = linspace(0.001, 0.007, 100);
lambda_range = linspace(0.05, 0.2, 100);
psi_range = linspace(0.1, 0.5, 100);
phi_range = linspace(0.01, 0.02, 100);

% Set your initial conditionshere
initial_conditions = [B0, N0, P0, E0, T0]; 

for i = 1:length(phi2_range)
    [B_eq, ~, ~, ~, ~] = simulate_system(phi2_range(i), lambda, psi, phi, initial_conditions);
    B_eq_values{i} = B_eq;
end

%----------------------- For forest resource B(t)-------------------------------------

p_val=32;
e_val1=10; 
e_val2=50;

figure('DefaultAxesFontSize',14)
set(gcf,'color','w');
plot(phi2_range(1:p_val),B_eq_values{1,e_val1}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','b',...
    'MarkerSize',20)
hold on;
plot(phi2_range(1:p_val),B_eq_values{1,e_val2}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','r',...
    'MarkerSize',20)
xlabel('\phi_2','fontsize',24)
ylabel('Forest Resource B(t)','fontsize',24)


%----------------------- For human population N(t)-------------------------------------

for i = 1:length(phi2_range)
    [~, N_eq, ~, ~, ~] = simulate_system(phi2_range(i), lambda, psi, phi, initial_conditions);
    N_eq_values{i} = N_eq;
end

p_val=32;
e_val1=10; 
e_val2=50;


figure('DefaultAxesFontSize',14)
set(gcf,'color','w');
plot(phi2_range(1:p_val),N_eq_values{1,e_val1}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','b',...
    'MarkerSize',20)
hold on;
plot(phi2_range(1:p_val),N_eq_values{1,e_val2}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','r',...
    'MarkerSize',20)
xlabel('\phi_2','fontsize',24)
ylabel('Human population N(t)','fontsize',24)

%----------------------- For Economic efforts E(t)-------------------------------------

for i = 1:length(phi2_range)
    [~, ~, ~, E_eq, ~] = simulate_system(phi2_range(i), lambda, psi, phi, initial_conditions);
    E_eq_values{i} = E_eq;
end

p_val=32;
e_val1=10; 
e_val2=50;

figure(2)
figure('DefaultAxesFontSize',14)
set(gcf,'color','w');
plot(phi2_range(1:p_val),E_eq_values{1,e_val1}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','b',...
    'MarkerSize',20)
hold on;
plot(phi2_range(1:p_val),E_eq_values{1,e_val2}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','r',...
    'MarkerSize',20)
xlabel('\phi_2','fontsize',24)
ylabel('Human population N(t)','fontsize',24)

%----------------------- For Population pressure P(t)-------------------------------------

for i = 1:length(phi2_range)
    [~, ~, ~, E_eq, ~] = simulate_system(phi2_range(i), lambda, psi, phi, initial_conditions);
    E_eq_values{i} = E_eq;
end

p_val=32;
e_val1=10; 
e_val2=50;

figure(2)
figure('DefaultAxesFontSize',14)
set(gcf,'color','w');
plot(phi2_range(1:p_val),E_eq_values{1,e_val1}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','b',...
    'MarkerSize',20)
hold on;
plot(phi2_range(1:p_val),E_eq_values{1,e_val2}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','r',...
    'MarkerSize',20)
xlabel('\phi_2','fontsize',24)
ylabel('Human population N(t)','fontsize',24)

%----------------------- For technological efforts T(t)-------------------------------------

for i = 1:length(phi2_range)
    [~, ~, ~, ~, T_eq] = simulate_system(phi2_range(i), lambda, psi, phi, initial_conditions);
    E_eq_values{i} = E_eq;
end

p_val=32;
e_val1=10; 
e_val2=50;

figure(2)
figure('DefaultAxesFontSize',14)
set(gcf,'color','w');
plot(phi2_range(1:p_val),E_eq_values{1,e_val1}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','b',...
    'MarkerSize',20)
hold on;
plot(phi2_range(1:p_val),E_eq_values{1,e_val2}(1:p_val),'.','MarkerIndices',...
    (1:length(phi2_range)),...
    'LineWidth',3,...
    'MarkerEdgeColor','r',...
    'MarkerSize',20)
xlabel('\phi_2','fontsize',24)
ylabel('Human population N(t)','fontsize',24)

%%
for i = 1:length(lambda_range)
    [~, N_eq, ~, ~, ~] = simulate_system(phi2, lambda_range(i), psi, phi, initial_conditions);
    N_eq_values{i} = N_eq;
end


%%

for i = 1:length(psi_range)
    [~, ~, P_eq, ~, ~] = simulate_system(phi2, lambda, psi_range(i), phi, initial_conditions);
    P_eq_values{i} = P_eq;
end

for i = 1:length(phi_range)
    [~, ~, ~, E_eq, ~] = simulate_system(phi2, lambda, psi, phi_range(i), initial_conditions);
    E_eq_values{i} = E_eq;
end

% Plot bifurcation diagrams
figure;
subplot(2, 2, 1);
plot(phi2_range, B_eq_values, '-o');
xlabel('phi2');
ylabel('B equilibrium');

subplot(2, 2, 2);
plot(lambda_range, N_eq_values, '-o');
xlabel('lambda');
ylabel('N equilibrium');

subplot(2, 2, 3);
plot(psi_range, P_eq_values, '-o');
xlabel('psi');
ylabel('P equilibrium');

subplot(2, 2, 4);
plot(phi_range, E_eq_values, '-o');
xlabel('phi');
ylabel('E equilibrium');


subplot(2, 1, 2);
plot3(Y(:, 1), Y(:, 2), Y(:, 3));
xlabel('B');
ylabel('N');
zlabel('P');
title('Phase Space: B-N-P');
grid on;
%%
plot3(t1, Y1(:, 1), Y1(:, 2), 'r', 'LineWidth', 1.5);
xlabel('B');
ylabel('N');
zlabel('P');
title('Phase Space: B-N-P');
grid on;
hold on;
plot3(t1, Y1(:, 2), Y1(:, 3),'g', 'LineWidth', 1.5);
hold on;
plot3(t2, Y2(:, 2), Y2(:, 3),'b', 'LineWidth', 1.5);


%%

% Compute equilibrium values
B_star = 7.9360; 
N_star = 100.6348; 
P_star = 5.3700; 
E_star = 10.7400;
T_star = 14.0213;

% Define time span
tspan = [0 100];

% Define initial conditions
B0 = B_star + 5;
N0 = N_star + 5;
P0 = P_star + 5;
E0 = E_star + 5;
T0 = T_star + 5;

% Define the system of equations
ode = @(t, y) [
    s * y(1) * (1 - y(1)/L) - alpha * y(1) * y(2) - lambda2 * y(1)^2 * y(3) + phi1 * y(1) * y(5) + phi2 * y(1)^2 * y(5);
    r * y(2) * (1 - y(2)/K) + Pi * alpha * y(1) * y(2);
    lambda * y(2) - lambda0 * y(3) - lambda1 * y(3) * y(4);
    psi * y(3) - psi0 * y(4);
    Pi * (L - y(1)) - phi0 * y(5)
];

% Solve the system of equations
[t, Y] = ode45(ode, tspan, [B0, N0, P0, E0, T0]);


% Plot the trajectories in the B-N-E space
plot(t, Y(:,1), 'r', 'LineWidth', 1.5);hold on; 
plot(t, Y(:,2), 'g', 'LineWidth', 1.5); hold on; 
plot(t, Y(:,3), 'g', 'LineWidth', 1.5);hold on;
plot(t, Y(:,4), 'b', 'LineWidth', 1.5);hold on; 


figure;
plot3(Y(:,1), Y(:,2), Y(:,4), 'b', 'LineWidth', 1.5);
hold on;
plot3(B_star, N_star, E_star, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('B'); ylabel('N'); zlabel('E');
title('Trajectories in B-N-E Space');
grid on;
hold off;

% Plot the trajectories in the B-N-T space
figure;
plot3(Y(:,1), Y(:,2), Y(:,5), 'b', 'LineWidth', 1.5);
hold on;
plot3(B_star, N_star, T_star, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('B'); ylabel('N'); zlabel('T');
title('Trajectories in B-N-T Space');
grid on;
hold on;

% Plot the trajectories in the B-P-E space
plot3(Y(:,1), Y(:,3), Y(:,4), 'b', 'LineWidth', 1.5);
hold on;
plot3(B_star, P_star, E_star, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('B'); ylabel('P'); zlabel('E');
title('Trajectories in B-P-E Space');
grid on;
hold off;


%%
% Define a meshgrid for B, P, and E
phi= linspace(0,0.04,length(Y));
% Equation for population growth (B)
for i=1:length(Y)
Bx(i)=s*Y(i:i,1:1)*(1-Y(i:i,1:1)/L) - alpha*Y(i:i,1:1)*Y(i:i,2:2) - lambda2*(Y(i:i,1:1))^2*Y(i:i,3:3) + phi1*(Y(i:i,1:1))*(Y(i:i,5:5)) + phi2*(Y(i:i,1:1))^2*Y(i:i,5:5);
end
% Equation for forest resources (N)
for i=1:length(Y)
Nx(i)= r*Y(i:i,2:2)*(1-(Y(i:i,2:2)/K))+Pi*alpha*(Y(i:i,1:1))*(Y(i:i,2:2));
end
% Equation for forest resources (P)
for i=1:length(Y)
Px(i)= lambda*Y(i:i,2:2)-lambda0*(Y(i:i,3:3))-lambda1*(Y(i:i,3:3))*(Y(i:i,4:4));
end

plot(phi,Bx,'.','MarkerIndices',...
    (1:length(B_mesh)),...
    'LineWidth',1.5,...
    'MarkerEdgeColor','b')
hold on;
plot(phi,Nx,'.','MarkerIndices',...
    (1:length(B_mesh)),...
    'LineWidth',1.5,...
    'MarkerEdgeColor','r')
hold on;
plot(phi,Px,'.','MarkerIndices',...
    (1:length(B_mesh)),...
    'LineWidth',1.5,...
    'MarkerEdgeColor','g')


plot3(Bx,Nx,Px, 'b', 'LineWidth', 1.5);
grid on;

plot(fx,phi,'.','MarkerIndices',...
    (1:length(B_mesh)),...
    'LineWidth',1.5,...
    'MarkerEdgeColor','b')

B_mesh  = meshgrid(fx);

N_m=10;
B_mesh  = meshgrid(linspace(0, 15, 100));
P_mesh  = meshgrid(linspace(0, (lambda/lambda0)*N_m, 100));
E_mesh  = meshgrid(linspace(0, (lambda*psi/(lambda0*psi0))*N_m, 100));


% Compute the stability conditions
stability_condition_1 = lambda2^2 * L^2 < lambda0 * (s/L + lambda2 * P_mesh - phi2 * T_star);
stability_condition_2 = phi2^2 * L^2 < 2 * (phi0 * phi1 / phi) * (s/L + lambda2 * P_mesh - phi2 * T_star);
stability_condition_3 = lambda^2 < 2 * (r * lambda0 / (pi * K));

% Plot the stability regions
figure;

surf(B_mesh,P_mesh,E_mesh,double(stability_condition_1))
hold on;
surf(B_mesh, P_mesh, E_mesh, double(stability_condition_2));
hold on;
surf(B_mesh, P_mesh, E_mesh, double(stability_condition_3));
xlabel('B'); ylabel('P'); zlabel('E');
title('Global Stability Regions in B-P-E Space');
grid on;
legend('Stability Condition 1', 'Stability Condition 2', 'Stability Condition 3');



