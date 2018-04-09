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
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Size of Theta1 is 25 x 401 and Theta2 is 10 x 26
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
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

%=========Computing the hypothesis function
% We need to add 1's to the first column of a_1 due to the fact that we have to set x0=1 to fit
% our model: x0w0 +x1w1 +... xNwN
a_1=[ones(m,1) X]; % 5000*401
z_2= a_1*Theta1';  % 5000*25 
a_2= [ones(m,1) sigmoid(z_2)]; % 5000 *26
z_3= a_2*Theta2'; %5000*10
a_3=sigmoid(z_3); %5000*10
hypo=a_3; %size is 5000*10



%===========Computing cost function with regularized parameters
penalty=(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2,2))...
 + sum(sum(Theta2(:,2:end).^2,2)));
 
%==========Computing the cost function
% Mapping the value of y, which is between 1..K, to the value in the row.
% If y(5)=7, it means that 5 is equal to the 5th training set, 3 is that value 
% of output. Mapping this case study, we set third element of 5th row to 1
% , whereas the rest of elements will stay 0's.
% Size of Y is 5000*10

Y=zeros(m, num_labels);
temp=eye(num_labels);
for i=1:m
  Y(i,:)=temp(y(i),:);
endfor %5000*10
J=(1/m)*sum(sum((-Y).*log(hypo)-(1-Y).*log(1-hypo),2))+penalty;

% Mapping the value of y, which is between 1..K, to the value in the row.
% If y(5)=7, it means that 5 is equal to the 5th training set, 3 is that value 
% of output. Mapping this case study, we set third element of 5th row to 1
% , whereas the rest of elements will stay 0's.
% Size of Y is 5000*10



%=========Backpropagation algorilthm implementation
% Size of Theta1 is 25 x 401 and Theta2 is 10 x 26

delta3=hypo.-Y; %5000*10

delta2=(sigma3*Theta2).*sigmoidGradient([ones(size(z_2,1),1) z_2]); %size is 5000*26
delta2=sigma2(:,2:end); % remove the first column of 1's which is...
                        % corresponding to sigma0.

Delta2=sigma3'*a_2; %Partial derivative with size is 10*26 
Delta1=sigma2'*a_1; %Partial derivative with size is 26*401

Theta1Reg=(lambda/m).*[zeros(size(Theta1,1),1) Theta1(:,2:end)]; % First column must be set at 0's because we don't update the theta_0...
Theta2Reg=(lambda/m).*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

Theta1_grad=Delta1./m + Theta1Reg; %Gradient descent of theta1 with regularization
Theta2_grad=Delta2./m + Theta2Reg; %Gradient descent of theta2 with regularization

% Please note that these computation above only reutrn gradient descent of Theta1 or Theta2
% Theta will be updated  later on. Ex: Theta1=Theta1-anpha*Theta1_grad. This computation will be taken
% in another place.

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
