function [Y,Alpha] = NObSP_Tot(X,Xsv,m,Mdl)

% Function to decompose the output of a model in the partial contributions
% given by the input data. The input and output of the model are defined
% as follows:
%
% X: Input to the model, the dimensions are Nx(dxm) where N is the number
% of observations, d is the number of input regressors, and m y is the number
% of model delays.
% Xsv: Support vectors for the model.
% Mdl: Is th trained Model using SVM.
% Y: is a matrix of size Nxd, where each column contains the nonlinear
% contributions of each input regressor on the estimated output.

out = predict(Mdl,X); % Finding the predicted output of the model
X = (X-Mdl.Mu)./Mdl.Sigma; % Preprocessing the input data

[N,d] = size(X); % Computing the dimensions for the input data
reg = d/m; % computing the number of regressors
n_sv = size(Xsv,1); % Computing the size of the support vectors.

input = cell(reg,1); % Initializing a cell structure for the input matrices
comp_input = cell(reg,1); % Initializing a cell structure for complementing the input matrices.
K_input = cell(reg,1); % Initializing a cell structure for the kernel matrices of the input
K_comp_input = cell(reg,1); % Initializing a cell structure for the kernel matrices of the complement of the input
P = cell(reg,1); % Initializing a cell structure for the projection matrices
Y = zeros(N,reg); % Initializing the output matrix
Alpha = zeros(n_sv,reg); % Initializing the output matrix

Mc1 = eye(N)-ones(N,1)*ones(1,N)/N; % Left centering matrix
Mc2 = eye(n_sv)-ones(n_sv,1)*ones(1,n_sv)/n_sv; % Right centering matrix
dis_X = diag(X*X')-2*X*Xsv'+ones(N,1)*(diag(Xsv*Xsv'))'; % Computting the euclidean distances between input data and support vectors
K = exp(-1*dis_X./Mdl.KernelParameters.Scale); % Calculating the kernel matrix
Proyection_K = pinv(K'*K)*K'; % Computing the proyection matrix to define the alpha parameters

%% Decomposing the output.

for i = 1 : reg
    % preparing the input regressors and the complement matrices to compute the projection
    input{i} = zeros(N,d);
    input{i}(:,(i-1)*m+1:i*m) = X(:,(i-1)*m+1:i*m);
    comp_input{i} = X;
    comp_input{i}(:,(i-1)*m+1:i*m) = zeros(N,m);
    
    % Computing the distances to calculate the kernel matrices.
    dis_input = diag(input{i}*input{i}')-2*input{i}*Xsv'+ones(N,1)*(diag(Xsv*Xsv'))';
    dis_comp_input = diag(comp_input{i}*comp_input{i}')-2*comp_input{i}*Xsv'+ones(N,1)*(diag(Xsv*Xsv'))';
    
    % Computing the kernel matrices
    K_input{i} = Mc1*exp(-1*dis_input./Mdl.KernelParameters.Scale)*Mc2;
    K_comp_input{i} = Mc1*exp(-1*dis_comp_input./Mdl.KernelParameters.Scale)*Mc2;

    % Computing the projections
    P_comp_input = K_comp_input{i}*pinv((K_comp_input{i}'*K_comp_input{i}))*K_comp_input{i}';
    Q_comp_input = eye(N) - P_comp_input;
    P{i} = K_input{i}*pinv((K_input{i}'*Q_comp_input*K_input{i}))*K_input{i}'*Q_comp_input;

    % Computing the proyections
    Y(:,i) = P{i}*(out-mean(out));
    Alpha(:,i) = Proyection_K*Y(:,i);
end
