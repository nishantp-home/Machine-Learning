function myPlotData(X,y)
  figure; hold on;
  negative = find(y==0);
  positive = find(y==1);
  plot(X(positive, 1), X(positive, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
  plot(X(negative, 1), X(negative, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  hold on;
  % Labels and Legend
  xlabel('Exam 1 score')
  ylabel('Exam 2 score')

  % Specified in plot order
  legend('Admitted', 'Not admitted')
  hold off;
end


