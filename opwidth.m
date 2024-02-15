function[b]=opwidth(A,B)

% OPWIDTH: computes the operator bandwith of M = ∑ Ak ⨂ Bk.
%
% b = OPWIDTH(A,B) returns max{b(Ak) + b(Bk)} where b(.) denotes the
% bandwidth. The factor matrices may be stored along the pages of third
% order tensors A and B or as elements of cell arrays.

A=convert(A, 'cell');
B=convert(B, 'cell');

bA=cellfun(@bandwidth, A, 'UniformOutput', true);
bB=cellfun(@bandwidth, B, 'UniformOutput', true);

b=max(bA+bB);
end