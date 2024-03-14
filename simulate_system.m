function [B_eq, N_eq, P_eq, E_eq, T_eq] = simulate_system(phi2, lambda, psi, phi, initial_conditions)
  % Initializaing parameters 
    
% Define parameter values
r = 0.1;    % Intrinsic growth rate
K = 100;    % Carrying capacity
Pi =0.01 ;   % Proportionality constant for population growth due to forest resources
alpha =0.008 ;    % Depletion rate coefficient of forest resources due to population
lambda = lambda;   % Growth rate coefficient of population pressure
lambda0 = 0.8 ;  % Natural depletion rate coefficient of population pressure
lambda1 =0.1 ;  % Depletion rate coefficient of population pressure due to economic efforts
psi = psi ;  % Growth rate coefficient of economic efforts
psi0 = 0.1 ; % Depletion rate coefficient of economic efforts
phi =phi ;  % Implementation rate coefficient of technological efforts
phi0 =0.03 ; % Depletion rate coefficient of technological efforts
s =0.6 ;    % Forest resources growth rate
L =50 ;    % Carrying capacity of forest resources
lambda2 =0.0011 ;  % Population pressure growth rate coefficient due to forest resources
phi1 =0.02 ; % Implementation rate coefficient of technological efforts on forest resources
phi2 =phi2 ; % Implementation rate coefficient of technological efforts on forest resources squared
% Time span = 150 years

t_start=1;
t_end=150;
tspan = [t_start, t_end];

 ode = @(t, y) [s*y(1)*(1-y(1)/L) - alpha*y(1)*y(2) - lambda2*y(1)^2*y(3) + phi1*y(1)*y(5) + phi2*y(1)^2*y(5);
               r*y(2)*(1-y(2)/K) + pi*alpha*y(1)*y(2);
               lambda*y(2) - lambda0*y(3) - lambda1*y(3)*y(4);
               psi*y(3) - psi0*y(4);
               phi*(L-y(1)) - phi0*y(5)];
% Initial conditions
B0=initial_conditions(1);
N0=initial_conditions(1);
P0=initial_conditions(1);
E0=initial_conditions(1);
T0=initial_conditions(1);
           
           [t, Y] = ode45(ode, tspan, [B0, N0, P0, E0, T0]); 
           B_eq=Y(:,1);
           N_eq=Y(:,2);
           P_eq=Y(:,3);
           E_eq=Y(:,4);
           T_eq=Y(:,5);

end