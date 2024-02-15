function[C_out]=gemm(A,B,C)

% GEMM: Implementation of the GEMM (General Matrix Multiply) operation
% C + AB.
%
% Y = GEMM(A,B,C) returns C + AB.

C_out=C+A*B;
end