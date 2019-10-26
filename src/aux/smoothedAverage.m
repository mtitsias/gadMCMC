function mF = smoothedAverage(F, Wind)
%
%

mF = zeros(1,length(F)); 
for n=1:length(F)
   st = n-Wind+1;
   st(st<1)=1;
   mF(n) = mean(F(st:n)); 
end
