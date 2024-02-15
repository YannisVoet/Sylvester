function[Y]=linOp(A,B,X,n,m)

% LINOP: computes ∑ Bk*X*Ak'.
%
% Y = LINOP(A,B,X,n,m) computes ∑ Bk*X*Ak' for cell arrays A and B storing 
% the factor matrices Ak and Bk of size n and m, respectively.

Y=sparse(m,n);
for k=1:length(A)
    Y=Y+B{k}*X*A{k}';
end
end