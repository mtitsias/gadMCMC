function [accept, A] = MHstep(newLogPx, oldLogPx, newLogProp, oldLogProp)
%
%Desctiption:  The general Metropolis-Hastings step 
%

A = newLogPx + oldLogProp - oldLogPx - newLogProp;
A = min(1, exp(A));

accept = 0;
u = rand;
if u < A
   accept = 1;
end

          