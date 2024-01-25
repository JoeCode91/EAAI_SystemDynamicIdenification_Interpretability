function [input,output] = prepareData(X,y,m)

% This function prepares the data for the SVM regression model. Here we
% produce an input matrix to the model, where each column of the input
% matrix X is hankelized, using order m; it also produces an output vector
% or matrix, with columns from the m+1 sample of the vector/matrix y. The
% description of the inputs for the function is as follows:
%
% X: Matrix with the input regressor variables, dimensions Nxd.
% y: Observed output of the model; it can be a vector or a matrix with
% dimensions Nx1 or Nxp.
% input: Hankelized version of the input X, dimensions are (N-m+1)x(d*m).
% output: Corresponding output for each input observation. Dimensions are
% (N-m+1)x1 or (N-m+1)xp

%% Hankelizatiopn to create the output.

[N,d] = size(X);
input = zeros(N-m+1,d*m);

for i = 1:d
    aux = hankel(X(1:end-m+1,i),X(end-m+1:end,i));
    input(:,(i-1)*m+1:i*m) = aux;
end

output = y(m:end,:);
