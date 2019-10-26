function [X,T,Xtest,TtestTrue] = loadMnist(train_data_per_class)

load  mnist_all.mat;


flag = 0;
if nargin == 1
   flag = 1;
end

T = [];
X = [];
K = 10; 
TtestTrue = []; 
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
for j=1:10
% 
    s = ['train' num2str(j-1)];
    Xtmp = eval(s);
    if flag == 1
       Xtmp = Xtmp(1:train_data_per_class,:);
    end
    Xtmp = double(Xtmp);   
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j), K); 
    Ttmp(:,j) = 1; 
    X = [X; Xtmp]; 
    T = [T; Ttmp]; 
    
    s = ['test' num2str(j-1)];
    Xtmp = eval(s); 
    Xtmp = double(Xtmp);
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j), K); 
    Ttmp(:,j) = 1; 
    Xtest = [Xtest; Xtmp]; 
    TtestTrue = [TtestTrue; Ttmp];    
%    
end
X = X/255; 
Xtest = Xtest/255; 
