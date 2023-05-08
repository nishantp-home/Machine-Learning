function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));        % (25 x 401)

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));              % (10 x 26)

% Setup some useful variables
    m = size(X, 1);       % number of training samples
         
% You need to return the following variables correctly 
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    X_wb = [ones(m,1), X];              % Feature matrix with bias (m = 5000) matrix size (5000 x 401)
    A = sigmoid(X_wb*Theta1');          % Activation matrix size (5000 x 25)
    A = [ones(m,1) A];                  % Added bias matrix size (5000 x 26)
    B = sigmoid(A*Theta2');             % Cost matrix size       (5000 x 10)

%[val p]=max(B, [], 2);
    y_matrix = zeros(m,num_labels); 
    index = (y-1).*m + (1:m)';     % index position for the y_matrix (unrolled) to be set to one
    y_matrix(index) = 1;           % (5000 x 10) matrix where each row is a label vector with one 1 and rest zeros

    costMatrix = -(y_matrix.*log(B)+(1-y_matrix).*log(1-B))/m;     % element-wise computation of cost  matrix size (5000 x 10)

    sumcol = sum(costMatrix);   % summing up all the columns (iterating over num_labels) to get a column vector (5000 x 1)
% compute cost J
    J = sum(sumcol);            % summing up all the entries of the vector (iterating over m) (scalar value)

% regularization term
% Theta1 and Theta2 without bias
    Theta1wb = Theta1(:, 2:end); % matrix size (25 x 400)
    Theta2wb = Theta2(:, 2:end); % matrix size (10 x 25)

    J = J + (lambda/(2*m))*(sum(sum(Theta1wb.*Theta1wb))+...
                            sum(sum(Theta2wb.*Theta2wb)));  % adding regularization terms to J


% compute gradients
    for t = 1:m       % iterating over each sample vector 
        A1 = X_wb(t,:);    % (1 x 401)
        A1 = A1';          % (401 x 1)
        z2 = Theta1*A1;    % (25 x 1)
        A2 = sigmoid(z2);  % (25 x 1)
        A2 = [1; A2];      % (26 x 1)
        z3 = Theta2*A2;    % (10 x 1)
        A3 = sigmoid(z3);  % (10 x 1)
        y = y_matrix(t,:); % (1 x 10)
        y = y';            % (10 x 1)
        delta3 = A3 - y;   % (10 x 1)
        delta2 = (Theta2'*delta3).*(A2.*(1-A2));   % matrix size (26 x 1)
        delta2 = delta2(2:end);                    % matrix size (25 x 1)
        Theta1_grad = Theta1_grad + delta2*A1';    %
        Theta2_grad = Theta2_grad + delta3*A2';
    end;

    Theta1_grad = Theta1_grad/m;
    Theta2_grad = Theta2_grad/m;

    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);
    
    
    grad = [Theta1_grad(:) ; Theta2_grad(:)];     % unroll the gradients into single vector


end
