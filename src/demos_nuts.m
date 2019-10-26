% all nuts experiments
% you will need to download and place in appropritate folder the 
% the  MATLAB NUTS implementation from here https://github.com/aki-nishimura/NUTS-matlab
clear;
randn('seed', 0);
rand('seed', 0);

% how many times to repeat the experiments 
Repeats = 10;
for r=1:Repeats
  demo_binaryclassification_nuts(r); 
end

for r=1:Repeats
  demo_radfordnealexample_nuts(r);
end

Repeats = 5;
for r=1:Repeats
  demo_mnist_nuts(r);
end

