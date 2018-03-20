function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

    
% Return true if z is a scalar or a vector, or false if z is actually a matrix (with m*n size of two dimensions)
if (isempty(z)) 
  for i=1:size(z)
     g(i)=1/(1+exp(-z(i)));
  endfor   
else  
  sizeMatrix=size(z);
  for i=1:sizeMatrix(1,1)
    for j=1:sizeMatrix(1,2)
           g(i,j)=1/(1+exp(-z(i,j)));
    endfor
  endfor
endif
% =============================================================

end
% The code to compute sigmoid function from the author of the course contains one line of code!!
% Again, I keep this to remind me later that how badly I came up sollution for the problem in these days.
