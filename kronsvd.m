function[Y,Z,varargout]=kronsvd(A,B,varargin)

% KRONSVD: Best Kronecker rank q approximation of a Kronecker rank r matrix.
%
% [Y,Z] = KRONSVD(A,B) computes the best Kronecker rank 1 approximation of 
% the Kronecker rank r matrix M = ∑ Ak ⨂ Bk. The factor matrices may be 
% stored either along the pages of third order tensors A and B or as 
% elements of cell arrays or in vectorized format as columns of matrices 
% A and B. By default, the factor matrices of the approximation are stored
% in the same format as the input.
%
% [Y,Z] = KRONSVD(A,B,q) computes the best Kronecker rank q approximation
% (with q ≤ r).
%
% [Y,Z] = KRONSVD(A,B,q,format) specifies the format of the output.
% Available choices are:
%   'tensor' - the factor matrices are stored along the pages of third
%              order tensors Y and Z. This format is only available for 
%              dense approximations.
%   'cell'   - the factor matrices are stored as elements of cell arrays.
%   'vect'   - the factor matrices are stored in vectorized format as
%              columns of the matrices Y and Z.
%
% [Y,Z,err] = KRONSVD(A,B,...) returns the low-rank approximation error in
% the Frobenius norm.
%
% References:
% [1] C. F. Van Loan and N. Pitsianis. Approximation with Kronecker products.
% In Linear Algebra for Large Scale and Real-Time Applications. Springer, 1993.
% [2] G. H. Golub and C. F. Van Loan. Matrix computations. JHU press, 2013.
%
% See also KRONALS.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('A');
Param.addRequired('B');
Param.addOptional('rank', 1, @(x) isscalar(x) & x > 0);
Param.addOptional('format', 'default', @(x) ismember(x,{'tensor','cell','vect'}));
Param.parse(A,B,varargin{:});

%% Retrieve parameters and check format
q=Param.Results.rank;
fout=Param.Results.format;

[A,sA,fA]=convert(A, 'vect');
[B,sB,fB]=convert(B, 'vect');

if ~isequal(fout, 'default')
    fA=fout;
    fB=fout;
end

if sA(3)~=sB(3)
    error('A and B must have the same number of factor matrices')
else
    r=sA(3);
end

%% Kronecker SVD

% QR decompositions
[QA,RA]=qr(A, 'econ');
[QB,RB]=qr(B, 'econ');

% Singular value decomposition
[U,S,V]=svd(full(RA*RB'));

Y=QA*U*sqrt(S);
Z=QB*V*sqrt(S);

Y=Y(:,1:q);
Z=Z(:,1:q);

S=diag(S);

if q<r
    varargout{1}=norm(S(q+1:end));
else
    varargout{1}=0;
end

Y=convert(Y, fA);
Z=convert(Z, fB);

end