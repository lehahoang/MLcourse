function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%sum()

%computing zSigmoid

% Calculating h theta X size is 5x4; theta size is 4x1 
% Size of hypo will be 5x1
% Size of y is 5x1
% Plz keep in mind that theta(1)=0 due to regularization
% or you will have no clue why the code is not working.
hypo=sigmoid(X*theta);
errorJ=y.*log(hypo)+log(1-hypo).*(1-y);
thetaReg=[0; theta(2:end,:)];
J=(-1/m)*sum(errorJ)+(lambda/2)*(1/m)*sum(thetaReg.^2);


%unregularized gradient
error=hypo-y; %size is 5x1
%size of X is 5x4
grad=grad+(1/m)*((X'*error)+lambda*thetaReg);

% =============================================================

grad =grad(:);

end
