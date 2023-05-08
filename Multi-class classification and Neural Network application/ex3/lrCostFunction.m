function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    m = length(y);  % number of training examples

    hypothesis = sigmoid(X*theta);

    % cost function
    J = -(sum(y.*log(hypothesis))+sum((1-y).*log(1-hypothesis)))/m + (lambda/(2*m))*(sum(theta(2:end).^2));

    %gradient of cost w.r.t theta vector
    grad = (X'*(hypothesis-y))/m;

    % not using theta0 for regularization
    temp = theta;
    temp(1) = 0;
    
    % adding regularization term in gradient computation
    grad = grad + (lambda/m)*temp;

end
