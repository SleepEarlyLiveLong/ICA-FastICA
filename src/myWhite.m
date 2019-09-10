function Y = myWhite(X,DIM)
%MYWHITE - The whitening function.
%   To calculate the white matex of input matrix X and 
%   the result after X being whitened. 
%   
%   Res = myWhite(X,DIM)
% 
%   Input - 
%   X: a N*M matrix containing M datas with N dimensions;
%   DIM: specifies a dimension DIM to arrange X.
%       DIM = 1: X(N*M)
%       DIM = 2: X(M*N)
%       DIM = otherwisw: error
%   Output - 
%   Y  : result matrix of X after being whitened;
%       Y.PCAW: PCA whiten result;
%       Y.ZCAW: ZCA whiten result.
% 
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

%% parameter test
if nargin < 2
    DIM = 1;
end
if DIM == 2
    X = X';
elseif DIM~=1 && DIM~=2
    error('Error! Parameter DIM should be either 1 or 2.');
end
[~,M] = size(X);

%% whitening
% step1 PCA pre-processing
X = X - repmat(mean(X,2),1,M);        % de-mean
C = 1/M*X*(X');                       % calculate cov(X), or: C = cov((X)')
[eigrnvector,eigenvalue] = eig(C);    % calculate eigenvalue, eigrnvector
% TEST NOW: eigrnvector*(eigrnvector)' should be identity matrix.
% step2 PCA whitening
if all(diag(eigenvalue))    % no zero eigenvalue
    Xpcaw = eigenvalue^(-1/2) * (eigrnvector)' * X;
else
    vari = 1./sqrt(diag(eigenvalue)+1e-5);
    Xpcaw = diag(vari) * (eigrnvector)' * X;
end
% Xpczw = (eigenvalue+diag(ones(size(X,1),1)*(1e-5)))^(-1/2)*(eigrnvector)'*X;    % 数据正则化
% step3 ZCA whitening
Xzcaw = eigrnvector*Xpcaw;
% TEST NOW: cov((Xpczw)') and cov((Xzcaw)') should be identity matrix.

%% result output
Y.PCAW = Xpcaw;
Y.ZCAW = Xzcaw;

end