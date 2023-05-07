function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
    m = length(y); % number of training examples
    n = size(theta);

    % J = 0;
    % grad = zeros(n);

    J = -(sum(y.*log(sigmoid(X*theta)))+sum((1-y).*log(1-sigmoid(X*theta))))/m + (lambda/(2*m))*(sum(theta.*theta)-theta(1)*theta(1));
    
    grad(1) = (sum((sigmoid(X*theta)-y).*X(:, 1)))/m;
    for i = 2:n
        grad(i) = (sum((sigmoid(X*theta)-y).*X(:, i)))/m + (lambda/m)*theta(i);
    end


end
