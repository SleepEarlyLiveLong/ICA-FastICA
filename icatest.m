% ica_test.m:
%   This file is used for testing the 
%   ICA(Independent Component Analysis) algorithm;
%   FastICA(Fast Independent Component Analysis) algorithm;
%   Whitening algorithm.
%
%   Copyright (c) 2018 CHEN Tianyang
%   more info contact: tychen@whu.edu.cn

clear;close;
%% ICA test
% -------------------在读入信号是纯净的非混叠信号时-----
[s1,fs] = audioread('voice\sound1.wav');
[s2,~] = audioread('voice\sound2.wav');
s1 = s1(:)';
s2 = s2(:)';
s1 = s1/max(s1);
s2 = s2/max(s2);
L = min(length(s1),length(s2));
s1 = s1(1:L);
s2 = s2(1:L);

% 语音混叠
H = rand(5,2)*[s1;s2];
mx = mean(H,2);
% c = cov(H');
H = H - mx*ones(1,L);   % subtract means from mixes 字段减去均值，中心化
% wz=2*inv(sqrtm(c));     % get decorrelating matrix
% X = wz*H;               % decorrelate mixes so cov(X')=4*eye(N) 
X=H;

% -------------------在直接读入已混叠的信号时-------
% [s1,fs] = audioread('c1_mixed1.wav');
% [s2,~] = audioread('c1_mixed2.wav');
% s1 = s1/max(s1);
% s2 = s2/max(s2);
% X=[s1';s2'];
% X=X-repmat(mean(X,2),1,size(X,2));
% figure;plot(s1);
% figure;plot(s2);
% -------------------over--------------------------

% 信号时序扰乱
% [N,P]=size(X);                       % P=48000, N=2, for example
% permute=randperm(P);                 % generate a permutation vector
% X_mixed=X(:,permute);                % time-scrambled inputs for stationarity 时序扰乱
%-------------------------------
tic;
S = myICA(X(1:2,:));
% S = myICA2(X,2);
% S_mixed = myICA(X_mixed);
toc
% 如输入信号时序扰乱，则输出需将序列理顺
% S_unmixed = zeros(N,P);
% for i = 1:P
%     sample = permute(1,i);
%     S_unmixed(:,sample) = S_mixed(:,i);
% end
%-------------------------------
sound(X(1,:),fs);
sound(X(2,:),fs);
sound(S(1,:),fs);
sound(S(2,:),fs);
sound(S_unmixed(1,:),fs);
sound(S_unmixed(2,:),fs);
sound(sum(S),fs);

%% FastICA test
close;clear;
% [s1,fs] = audioread('c1_mixed1.wav');
% s2 = audioread('c1_mixed2.wav');

[s1,fs] = audioread('voice\sound1.wav');
s2 = audioread('voice\sound2.wav');
s3 = audioread('voice\sound3.wav');
s4 = audioread('voice\sound4.wav');

s1 = s1/max(s1);
s2 = s2/max(s2);
s3 = s3/max(s3);
s4 = s4/max(s4);

figure;plot(s1);
figure;plot(s2);
figure;plot(s3);
figure;plot(s4);

S=[s1,s2,s3,s4]';   %信号组成4*N
[N,P] = size(S);
t = 1/fs:1/fs:P/fs;
A=rand(4);
% A=[0.8 0.9;0.9 0.8];
X=A*S;  %观察信号

%源信号波形图
figure(1);
subplot(4,1,1);plot(t,s1);axis([0 P/fs -1 1]);title('源信号');
subplot(4,1,2);plot(t,s2);axis([0 P/fs -1 1]);
subplot(4,1,3);plot(t,s3);axis([0 P/fs -1 1]);
subplot(4,1,4);plot(t,s4);axis([0 P/fs -1 1]);
xlabel('Time/s');
%观察信号(混合信号)波形图
figure(2);
subplot(4,1,1);plot(X(1,:));title('观察信号(混合信号)');
subplot(4,1,2);plot(X(2,:));
subplot(4,1,3);plot(X(3,:));
subplot(4,1,4);plot(X(4,:));

tic;
% Z=FastICA(S);
Z = lihao_myfastica(S);
toc

sound(S(1,:));
sound(S(2,:));
sound(S(3,:));
sound(S(4,:));

sound(X(1,:));
sound(X(2,:));
sound(X(3,:));
sound(X(4,:));

sound(Z(1,:));  % 旅行
sound(Z(2,:));  % 考生
sound(Z(3,:));  % 马克思
sound(Z(4,:));  % 张爱玲

%% white test
mu=[0,2];               %数学期望
sigma=[1 2;2,6];        %协方差矩阵
points=mvnrnd(mu,sigma,1000);  %生成50个样本
points = points';
figure;scatter(points(1,:),points(2,:),'r*');
axis equal;
Y = myWhite(points,1);
figure;scatter(Y.PCAW(1,:),Y.PCAW(2,:),'r*');hold on;
scatter(Y.ZCAW(1,:),Y.ZCAW(2,:),'y*');
axis equal;
cov(points')
cov(Y.PCAW')
cov(Y.ZCAW')

%%