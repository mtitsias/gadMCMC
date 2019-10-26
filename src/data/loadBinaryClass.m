function [X,t]  = loadBinaryClass(dataName)

  switch(dataName)
   case 'Caravan'
       %Load and prepare train & test data
       load('caravan.mat');
       % End column contains binary response data
       t=X(:,end);
       X(:,end) = [];
   case 'Australian'
       %Load and prepare train & test data
       load('australian.mat');
       t=X(:,end);
       X(:,end)=[];
   case 'German'
       %Load and prepare train & test data
       load('german.mat');
       t=X(:,end);
       X(:,end)=[];
       % German Credit - replace all 1s in t with 0s
       t(find(t==1)) = 0;
       % German Credit - replace all 2s in t with 1s
       t(find(t==2)) = 1;
   case 'Heart'
       %Load and prepare train & test data
       load('heart.mat');
       t=X(:,end);
       X(:,end)=[];
       % German Credit - replace all 1s in t with 0s
       t(find(t==1)) = 0;
       % German Credit - replace all 2s in t with 1s
       t(find(t==2)) = 1;
   case 'Pima'
       %Load and prepare train & test data
       load('pima.mat');
       t=X(:,end);
       X(:,end)=[];
   case 'Ripley'
       %Load and prepare train & test data
       load('ripley.mat');
       t=X(:,end);
       X(:,end)=[];
  end

  [n D] = size(X);

  % normalize the data so that the inputs have unit variance and zero mean
  meanX = mean(X);
  sqrtvarX = sqrt(var(X));
  X = X - repmat(meanX, n, 1);
  X = X./repmat(sqrtvarX, n, 1);

