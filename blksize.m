function[n, varargout]=blksize(M, varargin)

% BLKSIZE: Attempts to find the block partitioning of a sparse matrix M.
%
% n = BLKSIZE(M) finds the block partitioning n = (n1,n2,...,nd) of a 
% sparse symmetric matrix M.
%
% [n,b] = BLKSIZE(M) also returns the bandwidths b = (b1,b2,...,bd).
%
% [n,b,r] = BLKSIZE(M) also returns the blocksizes r = (r1,r2,...,1).

%% Set algorithm parameters

[m, n]=size(M);

Param = inputParser;
Param.addRequired('matrix', @(x) size(x,1)==size(x,2) & max(abs(x-0.5*(x+x')), [], 'all')<1e-5 & nnz(x)<m*n);
Param.parse(M, varargin{:});

%% Block partitioning algorithm
% Block diagonal matrices are not detected by the algorithm; e.g. block
% diagonal matrices of type Id ⨂ M are identified with (d*n1,n2,...,nd).

[I,J]=find(M);
V=abs(I-J);
% "Distance" matrix. The matrix B has the same sparsity as A with entries
% b_{ij} = |i-j|
B=sparse(I,J,V,m,m);

s=sort(unique(B));
incr=s(2:end)-s(1:end-1);
% Symmetrize with respect to diagonal for an easier identification of
% bandwidths.
incr=[flipud(incr); incr];

l=sort(unique(incr), 'descend');

% Dimension
d=length(l);
% Bandwidths
b=ones(d+1,1);

for k=1:d
    b(k+1)=sum(incr==l(k))/prod(b(1:k))+1;
end
b=b(2:d+1);
b=0.5*(b-1);
varargout{1}=b;

if d==1
    warning('No pattern was found.')
    n=m;
    varargout{2}=1;
    return
end

% Block partitioning
% The components of r and z store the sizes and bandwidths, respectively,
% of the blocks on each hierarchical level; i.e. 
%  r_{d} = 1 and r_{j} = n_{j+1}*r_{j+1} = n_{j+1}*...*n_{d}.
z=zeros(d,1);
r=ones(d,1);
n=ones(d,1);

% Initialization
z(end)=b(end);

for k=d-1:-1:1
    r(k)=l(k)+2*z(k+1);
    z(k)=b(k)*r(k)+z(k+1);
    n(k+1)=r(k)/r(k+1);
end
n(1)=m/r(1);

varargout{2}=r;
end