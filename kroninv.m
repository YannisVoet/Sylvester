function[C,D,varargout] = kroninv(A,B,varargin)

% KRONINV: Alternating least squares algorithm for finding a Kronecker
% rank 1 approximation of the inverse.
%
% [C,D] = KRONINV(A,B) computes factor matrices C and D such that C ⨂ D
% approximates the inverse of ∑ Ak ⨂ Bk. The factor matrices may be stored
% along the pages of third order tensors A and B or as elements of cell
% arrays.
%
% [C,D] = KRONINV(A,B,name,value) specifies optional name/value pair
% arguments.
%   'C0'        - initial guess for the matrix C.
%                 Default: C = sparsity(∑ Ak).
%   'D0'        - initial guess for the matrix D.
%                 Default: D = sparsity(∑ Bk).
%   'format'    - Output format for C and D specified as {fC, fD}.
%                 Available formats are 'tensor' and 'cell'.
%                 Default: same format as input.
%   'tolerance' - tolerance for the alternating least squares algorithm.
%                 Default: 1e-3.
%   'nitermax'  - maximum number of iterations.
%                 Default: 10.
%   'sparse'    - computes a sparse approximate inverse. This functionality
%                 is only available for A,B defined as cell arrays.
%                 Default: false.
%   'parallel'  - solves linear systems in parallel (for computing sparse 
%                 approximate inverses and if the parallel computing 
%                 toolbox is available).
%                 Default: false.
%
% [C,D,res] = KRONINV(A,B,...) returns the vector of residuals at each
% iteration.
%
% See also KRONINVQ.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('A');
Param.addRequired('B');
Param.addParameter('C0', 'auto', @(x) ismember(class(x),{'double','cell'}));
Param.addParameter('D0', 'auto', @(x) ismember(class(x),{'double','cell'}));
Param.addParameter('format', 'auto', @(x) all(ismember(x,{'tensor','cell'})));
Param.addParameter('tolerance', 1e-3, @(x) isscalar(x) & x > 0);
Param.addParameter('nitermax', 10, @(x) isscalar(x) & x > 0);
Param.addParameter('sparse', false, @islogical);
Param.addParameter('parallel', false, @islogical);
Param.parse(A,B,varargin{:});

%% Retrieve parameters and check format
C=Param.Results.C0;
D=Param.Results.D0;
fout=Param.Results.format;
tol=Param.Results.tolerance;
nitermax=Param.Results.nitermax;
sp=Param.Results.sparse;
par=Param.Results.parallel;

% Initialize solver options
Solver.sp=sp;
Solver.par=par;

% Check input format
[A,sA,fA]=convert(A, {'cell'});
[B,sB,fB]=convert(B, {'cell'});

if sA(1)~=sA(2) || sB(1)~=sB(2)
    error('Factor matrices must be square')
else
    n=sA(1);
    m=sB(1);
end

if sA(3)~=sB(3)
    error('A and B must have the same number of factor matrices')
else
    r=sA(3);
end

if isequal(C, 'auto')
    C=sparinv(A,1,'tensor');
else
    [C,sC]=convert(C, {'tensor'});

    if sC(1)~=sC(2)
        error('Factor matrices must be square')
    end

    if sA(1)~=sC(1)
        error('Factor matrices must have compatible size')
    end

    if sC(3)~=1
        error('C must be a matrix')
    end
end

if isequal(D, 'auto')
    D=sparinv(A,1,'tensor');
else
    [D,sD]=convert(D, {'tensor'});

    if sD(1)~=sD(2)
        error('Factor matrices must be square')
    end

    if sB(1)~=sD(1)
        error('Factor matrices must have compatible size')
    end

    if sD(3)~=1
        error('D must be a matrix')
    end
end

if isequal(fout, 'auto')
    fout={fA, fB};
end

%% Alternating Least Squares

% Convert A and B to convenient formats for fast computations
Am=cell2mat(reshape(A, [r 1]));
Bm=cell2mat(reshape(B, [r 1]));

At=cell2vect(A,'transpose');
Bt=cell2vect(B,'transpose');

% Initialization
res=zeros(nitermax,1);
niter=1;
residual=inf;

% Compute coefficients beta and delta
Va=reshape((Am*C)', [n^2, r]);
beta=Va'*Va;
delta=sum(Va(1:(n+1):n^2,:),1);

while niter <= nitermax && residual > tol

    % Form and solve normal equations
    M=Bm'*kron(beta, speye(m,m))*Bm;
    % Symplifying the computation of Bm'*kron(Va', speye(m,m))*Eb by
    % exploiting the structure of Eb
    RHS=reshape(sum(delta.*Bt,2), [m m]);

    % Solve system
    [D]=solvesyst(D,M,RHS,Solver);

    % Compute coefficients alpha and gamma
    Vb=reshape((Bm*D)', [m^2, r]);
    alpha=Vb'*Vb;
    gamma=sum(Vb(1:(m+1):m^2,:),1);

    % Form and solve normal equations
    M=Am'*kron(alpha, speye(n,n))*Am;
    % Symplifying the computation of Am'*kron(Vb', speye(n,n))*Ea by
    % exploiting the structure of Ea
    RHS=reshape(sum(gamma.*At,2), [n n]);

    % Solve system
    [C]=solvesyst(C,M,RHS,Solver);

    % Update coefficients beta and delta
    Va=reshape((Am*C)', [n^2, r]);
    beta=Va'*Va;
    delta=sum(Va(1:(n+1):n^2,:),1);

    % Compute residual
    residual=sqrt(n*m-2*dot(gamma, delta)+sum(alpha.*beta, 'all'));
    res(niter)=residual;
    niter=niter+1;
end

% Convert output
[C]=convert(C, fout(1));
[D]=convert(D, fout(2));

varargout{1}=res;
end