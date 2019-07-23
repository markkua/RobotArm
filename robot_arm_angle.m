function [alpha, beta, gamma] = robot_arm_angle(x,y,z)

%??????
du = pi/180;
radian = 180/pi;

%???????
thet = pi/9;
OA = 5;
AB = 20;
BC = 20;

syms aipha;
alpha = solve('tan(alpha) = y/x' , 'alpha');

Ax = OA .* cos(thet) .* cos(alpha);
Ay = OA .* cos(thet) .* sin(alpha);
Az = OA .* sin(thet);
AC = sqrt((x-Ax).^2 + (y-Ay).^2 + (z-Az).^2);

syms  AC gamma;
gamma = solve('BC.^2 + AB.^2 -AC.^2 = 2 .* BC .* AB .* cos(gamma)' , 'gamma');

syms gamma beta thet x y
beta = solve('BC .* sin(gamma - beta) + AB .* sin(beta) + OA .* cos(thet) = sqrt(x.^2 + y.^2)' ,...
    'beta');

%?????????
alpha = alpha .* radian;
gamma = gamma .* radian;
beta = beta .* radian;

[alpha , beta , gamma]



