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

%% ȥ��ֵ
[M,T]=size(X);   %��ȡ��������������������Ϊ�۲����ݵ���Ŀ������Ϊ��������
average=mean((X'))';  %��ֵ
for i=1:M
    X(i,:)=X(i,:)-average(i)*ones(1,T);
end

%% �׻�/��
Y = myWhite(X,1);
Z = Y.PCAW;
% Z=X;
%% ����
Maxcount=10000;  %����������
Critical=0.00001;  %�ж��Ƿ�����
m=M;            %��Ҫ���Ƶķ����ĸ���
W=rand(m);
for n=1:m
    WP=W(:,n);  %��ʼȨʸ�������⣩
    %Y=WP'*Z;
    %G=Y.^3;%GΪ�����Ժ�������ȡy^3��
    %GG=3*Y.^2��   %G�ĵ���
    count=0;
    LastWP=zeros(m,1);
    W(:,n)=W(:,n)/norm(W(:,n));         %��λ��һ������
    while (abs(WP-LastWP) & abs(WP+LastWP)) > Critical    %��������ֵͬʱ������������
        count=count+1;  %��������
        LastWP=WP;      %�ϴε�����ֵ
        %WP=1/T*Z*((LastWP'*Z).^3)'-3*LastWP;
        for i=1:m
            %����
            WP(i)=mean( Z(i,:).*(tanh((LastWP)'*Z)) )-(mean(1-(tanh((LastWP))'*Z).^2)).*LastWP(i);
        end
        WPP=zeros(m,1);     %ʩ����������
        for j=1:n-1
            WPP=WPP+(WP'*W(:,j))*W(:,j);
        end
        WP=WP-WPP;
        WP=WP/(norm(WP));
        
        if count==Maxcount
            fprintf('δ�ҵ���Ӧ���ź�');
            return;
        end
    end
    W(:,n)=WP;
end

%% �������
Z=W'*Z;

end
%% 