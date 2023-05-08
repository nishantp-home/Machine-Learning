function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

    m = length(y); % number of training examples

    J = (1/(2*m))*((X*theta-y)'*(X*theta-y) + lambda*(theta'*theta-theta(1)*theta(1)));
    
    % Not considering bias term in the regualrization 
    theta_n = theta;
    theta_n(1)=0;   % Setting bias term component to zero
    grad = (1/m)*((X'*(X*theta-y))+lambda*theta_n);

    grad = grad(:);

end
