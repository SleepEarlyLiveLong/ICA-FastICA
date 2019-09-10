function Z = FastICA(X)
%FASTICA - The FastICA(Fast Independent Component Analysis) algorithm.
%   To seperate independent signals from a mixed matrix X, the unmixed
%   signals are saved as rows of matrix Z.
%   Here are some useful reference material:
%   https://blog.csdn.net/zf_suan/article/details/53750455
%
%   Z = FastICA(X)
% 
%   Input - 
%   X: a N*M matrix with mixed signals containing M datas with N dimensions;
%   Output - 
%   Z: a N*M matrix with unmixed signals.

%% 去均值
[M,T]=size(X);   %获取输入矩阵的行列数，行数为观测数据的数目，列数为采样点数
average=mean((X'))';  %均值
for i=1:M
    X(i,:)=X(i,:)-average(i)*ones(1,T);
end

%% 白化/球化
Y = myWhite(X,1);
Z = Y.PCAW;
% Z=X;
%% 迭代
Maxcount=10000;  %最大迭代次数
Critical=0.00001;  %判断是否收敛
m=M;            %需要估计的分量的个数
W=rand(m);
for n=1:m
    WP=W(:,n);  %初始权矢量（任意）
    %Y=WP'*Z;
    %G=Y.^3;%G为非线性函数，可取y^3等
    %GG=3*Y.^2；   %G的导数
    count=0;
    LastWP=zeros(m,1);
    W(:,n)=W(:,n)/norm(W(:,n));         %单位化一列向量
    while (abs(WP-LastWP) & abs(WP+LastWP)) > Critical    %两个绝对值同时大于收敛条件
        count=count+1;  %迭代次数
        LastWP=WP;      %上次迭代的值
        %WP=1/T*Z*((LastWP'*Z).^3)'-3*LastWP;
        for i=1:m
            %更新
            WP(i)=mean( Z(i,:).*(tanh((LastWP)'*Z)) )-(mean(1-(tanh((LastWP))'*Z).^2)).*LastWP(i);
        end
        WPP=zeros(m,1);     %施密特正交化
        for j=1:n-1
            WPP=WPP+(WP'*W(:,j))*W(:,j);
        end
        WP=WP-WPP;
        WP=WP/(norm(WP));
        
        if count==Maxcount
            fprintf('未找到相应的信号');
            return;
        end
    end
    W(:,n)=WP;
end

%% 数据输出
Z=W'*Z;

end
%% 