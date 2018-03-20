function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% computing the sum of squared regularization paramenter
reg_para=(lambda/(2*m))*sum(theta().^2);

 % calculating cost function======
for i=1:m
    J=J-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta));
    endfor
J=J/m+reg_para;

% computing the gradient descent===========
% computing theta 0
for i=1:m
  grad(1)=grad(1)+(1/m)*X(i,1)*(sigmoid(X(i,:)*theta)-y(i));    
endfor
  
% computing theta 1 to 
for j=2:size(theta) 
  for i=1:m 
    grad(j)=grad(j)+ X(i,j)*(sigmoid(X(i,:)*theta)-y(i));
  endfor
    grad(j)=grad(j)/m+(lambda/m)*theta(j);
endfor





% =============================================================

end