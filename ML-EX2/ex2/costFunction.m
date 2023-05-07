function [J, grad] = costFunction(theta, X, y)
%   COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
    m = length(y); % number of training examples
    hypothesis = sigmoid(X*theta);

    J = -(sum(y.*log(hypothesis))+sum((1-y).*log(1-hypothesis)))/m;
    grad(1) = (sum((hypothesis-y).*X(:,1)))/m;
    grad(2) = (sum((hypothesis-y).*X(:,2)))/m;
    grad(3) = (sum((hypothesis-y).*X(:,3)))/m;

end
