function[Y]=trgsyl(P,R,S,T,F)

% TRGSYL: Standard algorithm for solving small-sized generalized Sylvester
% equations PYR' + SYT' = F where P, T are upper quasi-triangular and R, S
% are upper triangular.
%
% Y = TRGSYL(P,R,S,T,F) solves the generalized Sylvester equation 
% PYR' + SYT' = F in upper (quasi-)triangular form.
%
% References:
% [1] J. D. Gardiner, A. J. Laub, J. J. Amato, and C. B. Moler. Solution of 
% the Sylvester matrix equation AXB'+ CXD' = E. ACM Transactions on 
% Mathematical Software (TOMS), 1992.
% [2] R. H. Bartels and G. W. Stewart. Algorithm 432 [C2]: Solution of the 
% matrix equation AX + XB = C [F4]. Communications of the ACM, 1972.

% Solves a system of equations in block (quasi-)triangular form.
% Similar to the Bartels and Stewart algorithm [2] for standard Sylvester
% equations.
m=size(P,1);
n=size(R,1);

Y=zeros(m,n);
k=n;

% Backward substitution algorithm
while k>=1
    if k>1
        if T(k,k-1)==0 % Solve the system of equations for block k
            Y(:,k)=(R(k,k)*P+T(k,k)*S)\F(:,k);
            F(:,1:k-1)=F(:,1:k-1)-(R(1:k-1,k)'.*(P*Y(:,k))+T(1:k-1,k)'.*(S*Y(:,k)));
            k=k-1;

        else % Solve the system of equations for block k and k-1 simultaneously
            RHS=F(:,k-1:k);
            W=(kron(R(k-1:k,k-1:k),P)+kron(T(k-1:k,k-1:k),S))\RHS(:);
            Y(:,k-1:k)=reshape(W, m, 2);
            F(:,1:k-2)=F(:,1:k-2)-(R(1:k-2,k-1)'.*(P*Y(:,k-1))+T(1:k-2,k-1)'.*(S*Y(:,k-1)));
            F(:,1:k-2)=F(:,1:k-2)-(R(1:k-2,k)'.*(P*Y(:,k))+T(1:k-2,k)'.*(S*Y(:,k)));
            k=k-2;
        end
    else % It is the first row
        Y(:,k)=(R(k,k)*P+T(k,k)*S)\F(:,k);
        k=k-1;
    end
end
end