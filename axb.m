function[C]=axb(A,B,C,X)

% AXB: Implementation of the GEMM (General Matrix Multiply) operation
% C + AXB.
%
% Y = AXB(A,B,C,X) returns C + AXB.

C=C+A*X*B;
end