function[Y,Z,varargout]=kronals(A,B,varargin)

% KRONALS: Kronecker rank q approximation of a Kronecker rank r matrix
% using an alternating least squares algorithm.
%
% [Y,Z] = KRONALS(A,B) computes a Kronecker rank 1 approximation of the
% Kronecker rank r matrix M = ∑ Ak ⨂ Bk. The factor matrices may be stored
% either along the pages of third order tensors A and B or as elements of
% cell arrays or in vectorized format as columns of matrices A and B.
%
% [Y,Z] = KRONALS(A,B,q) computes a Kronecker rank q approximation of the
% Kronecker rank r matrix M = ∑ Ak ⨂ Bk.
%
% [Y,Z] = KRONALS(A,B,q,fout) specifies the output format. By default, the 
% factor matrices of the approximation are stored in the same format as the 
% input. Available choices are:
%   'tensor' - the factor matrices are stored along the pages of third
%              order tensors Y and Z. This format is only available for 
%              dense approximations.
%   'cell'   - the factor matrices are stored as elements of cell arrays.
%   'vect'   - the factor matrices are stored in vectorized format as
%              columns of the matrices Y and Z.
%
% [Y,Z] = KRONALS(A,B,q,fout,name,value) specifies optional name/value
% pair arguments.
%   'Z0'        - initial guesses for the factor matrices Zs.
%                 Default: Z_1 = I and Z_{s+1} is constructed from Z_{s} by
%                 appending a super and sub-diagonal of ones.
%   'tolerance' - tolerance for the alternating least squares algorithm.
%                 Default: 1e-1.
%   'nitermax'  - maximum number of iterations.
%                 Default: 10.
%
% [Y,Z,res] = KRONALS(A,B,...) returns the vector of residuals at each
% iteration.
%
% References:
% [1] C. F. Van Loan and N. Pitsianis. Approximation with Kronecker products.
% In Linear Algebra for Large Scale and Real-Time Applications. Springer, 1993.
% [2] G. H. Golub and C. F. Van Loan. Matrix computations. JHU press, 2013.
%
% See also KRONSVD.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('A');
Param.addRequired('B');
Param.addOptional('rank', 1, @(x) isscalar(x) & x > 0);
Param.addOptional('format', 'default', @(x) ismember(x,{'tensor','cell','vect'}));
Param.addParameter('Z0', 'auto', @(x) ismember(class(x),{'double','cell'}));
Param.addParameter('tolerance', 1e-1, @(x) isscalar(x) & x > 0);
Param.addParameter('nitermax', 10, @(x) isscalar(x) & x > 0);
Param.parse(A,B,varargin{:});

%% Retrieve parameters and check format
q=Param.Results.rank;
Z=Param.Results.Z0;
fout=Param.Results.format;
tol=Param.Results.tolerance;
nitermax=Param.Results.nitermax;

[A,sA,fA]=convert(A, 'vect');
[B,sB,fB]=convert(B, 'vect');

if sA(1)~=sA(2) || sB(1)~=sB(2)
    error('Factor matrices must be square')
end

if sA(3)~=sB(3)
    error('A and B must have the same number of factor matrices')
end

if ~isequal(fout, 'default')
    fA=fout;
    fB=fout;
end

if isequal(Z, 'auto')
    Z=cellfun(@(x) spdiags(ones(sB(1),2*x+1), -x:x, sB(1), sB(1)), num2cell(0:(q-1)), 'UniformOutput', false);
end

[Z,sZ]=convert(Z, 'vect');

if sZ(1)~=sZ(2)
    error('Factor matrices must be square')
end

if sB(1)~=sZ(1)
    error('Factor matrices in B and Z must have compatible size')
end

if sZ(3)~=q
    error('The number of factor matrices in Z must be equal to q')
end

%% Alternating least squares

% Initialization
res=zeros(nitermax,1);
niter=1;
residual=inf;

while niter <= nitermax && residual > tol
    Y=A*((B'*Z)/(Z'*Z));
    Z=B*((A'*Y)/(Y'*Y));

    % Compute residual
    residual=sqrt(sum((A'*A).*(B'*B), 'all')-2*sum((A'*Y).*(B'*Z), 'all')+sum((Y'*Y).*(Z'*Z), 'all'));
    res(niter)=residual;
    niter=niter+1;
end

varargout{1}=res;

Y=convert(Y, fA);
Z=convert(Z, fB);

end