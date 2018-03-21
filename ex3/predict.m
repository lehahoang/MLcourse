function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


% Size of Theta1 is 25 x 41
% Size of Theta2 is 10 x 26
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a_1 = [ones(m, 1) X];
z_2 = zeros(size(X,1),1);
z_3 = zeros(size(X,1),1); 

% First computation at hidden layer==================

z_2=a_1*Theta1';
a_2=[ones(m,1) sigmoid(z_2)];% Adding comlumn of 1's which is equivalent to first x0


% Final computation at output layer==================
z_3=a_2*Theta2';
a_3= sigmoid(z_3); % Must not add column of 1's because this is the output layer
[max_values indices]=max(a_3,[],2);

% output the maximum of the value in each row which is corresponding to the number indicated.
p=indices;










% =========================================================================


end
