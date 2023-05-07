%initialization
clear; close all; clc

data = load("ex2data1.txt");
X = data(:, [1:2]);
y = data(:, 3);

myPlotData(X,y);


[m, n] = size(X);
X = [ones(m,1), X];

initialTheta = zeros(n+1, 1)

% Compute and display initial cost and gradient
[cost, grad] = myCostFunction(initialTheta, X, y);
