function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
    figure; 
    hold on;

    % Find Indices of Positive and Negative Examples
    positive = find(y == 1); 
    negative = find(y == 0);

    % Plot Examples
    plot(X(positive, 1), X(positive, 2), 'k+','LineWidth', 1.5, 'MarkerSize', 5);
    plot(X(negative, 1), X(negative, 2), 'ko', 'MarkerFaceColor', 'red', 'MarkerSize', 5);

    hold off;

end
