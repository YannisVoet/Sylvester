function[C,D,varargout] = kroninvq(A,B,varargin)

% KRONINVQ: Alternating least squares algorithm for finding a Kronecker
% rank q approximation of the inverse.
%
% [C,D] = KRONINVQ(A,B) computes factor matrices C and D such that C ⨂ D
% approximates the inverse of ∑ Ak ⨂ Bk. The factor matrices may be stored
% along the pages of third order tensors A and B or as elements of cell
% arrays.
%
% [C,D] = KRONINVQ(A,B,q) computes factor matrices Cs and Ds for
% s = 1,...,q such that ∑ Cs ⨂ Ds approximates the inverse of ∑ Ak ⨂ Bk.
%
% [C,D] = KRONINVQ(A,B,q,name,value) specifies optional name/value pair
% arguments.
%   'C0'        - initial guesses for the factor matrices Cs.
%                 Default: Cs = sparsity((∑ Ak)^s) for s = 1,...,q.
%   'D0'        - initial guesses for the factor matrices Ds.
%                 Default: Ds = sparsity((∑ Bk)^s) for s = 1,...,q.
%   'format'    - Output format for C and D specified as {fC, fD}.
%                 Available formats are 'tensor'and 'cell'.
%                 Default: same format as input.
%   'tolerance' - tolerance for the alternating least squares algorithm.
%                 Default: 1e-3.
%   'nitermax'  - maximum number of iterations.
%                 Default: 10.
%   'sparse'    - computes a sparse approximate inverse.
%                 Default: false.
%   'parallel'  - solves linear systems in parallel (for computing sparse 
%                 approximate inverses and if the parallel computing 
%                 toolbox is available).
%                 Default: false.
%
% [C,D,res] = KRONINVQ(A,B,q,...) returns the vector of residuals at each
% iteration.
%
% See also KRONINV.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('A');
Param.addRequired('B');
Param.addOptional('rank', 1, @(x) isscalar(x) & x > 0);
Param.addParameter('C0', 'auto', @(x) ismember(class(x),{'double','cell'}));
Param.addParameter('D0', 'auto', @(x) ismember(class(x),{'double','cell'}));
Param.addParameter('format', 'auto', @(x) all(ismember(x,{'tensor','cell'})));
Param.addParameter('tolerance', 1e-3, @(x) isscalar(x) & x > 0);
Param.addParameter('nitermax', 10, @(x) isscalar(x) & x > 0);
Param.addParameter('sparse', false, @islogical);
Param.addParameter('parallel', false, @islogical);
Param.parse(A,B,varargin{:});

%% Retrieve parameters and check format
q=Param.Results.rank;
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
    C=sparinv(A,1:q,'cell');
else
    [C,sC]=convert(C, {'cell'});

    if sC(1)~=sC(2)
        error('Factor matrices must be square')
    end

    if sA(1)~=sC(1)
        error('Factor matrices must have compatible size')
    end

    if sC(3)~=q
        error('The number of factor matrices of C must be equal to q')
    end
end

if isequal(D, 'auto')
    D=sparinv(A,1:q,'cell');
else
    [D,sD]=convert(D, {'cell'});

    if sD(1)~=sD(2)
        error('Factor matrices must be square')
    end

    if sB(1)~=sD(1)
        error('Factor matrices must have compatible size')
    end

    if sD(3)~=q
        error('The number of factor matrices of D must be equal to q')
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
alpha=zeros(r,r,q,q);
beta=zeros(r,r,q,q);
gamma=zeros(r,q);
delta=zeros(r,q);

Va=cell(1,q);
Vb=cell(1,q);

res=zeros(nitermax,1);
niter=1;
residual=inf;

% Compute coefficients beta and delta
for i=1:q
    Va{i}=reshape((Am*C{i})', [n^2, r]);
    delta(:,i)=sum(Va{i}(1:(n+1):n^2,:),1);
end

for i=1:q
    for j=1:q
        beta(:,:,i,j)=Va{i}'*Va{j};
    end
end

while niter <= nitermax && residual > tol

    M=cell(q,q);
    RHS=cell(q,1);

    % Form and solve normal equations
    for i=1:q
        % Symplifying the computation of Bm'*kron(Va', speye(m,m))*Eb by
        % exploiting the structure of Eb
        RHS{i}=reshape(sum(delta(:,i)'.*Bt,2), [m m]);
    end

    for i=1:q
        for j=1:q
            M{i,j}=Bm'*kron(beta(:,:,i,j), speye(m,m))*Bm;
        end
    end

    M=cell2mat(M);
    RHS=cell2mat(RHS);
    D=cat(1, D{:});

    % Solve system
    [D]=solvesyst(D,M,RHS,Solver);

    % Reshape D to cell array
    D=mat2cell(D, m*ones(q,1), m);

    M=cell(q,q);
    RHS=cell(q,1);

    % Form and solve normal equations
    for i=1:q
        Vb{i}=reshape((Bm*D{i})', [m^2, r]);
        % Symplifying the computation of Am'*kron(Vb', speye(n,n))*Ea by
        % exploiting the structure of Ea
        gamma(:,i)=sum(Vb{i}(1:(m+1):m^2,:),1);
        RHS{i}=reshape(sum(gamma(:,i)'.*At,2), [n n]);
    end

    for i=1:q
        for j=1:q
            alpha(:,:,i,j)=Vb{i}'*Vb{j};
            M{i,j}=Am'*kron(alpha(:,:,i,j), speye(n,n))*Am;
        end
    end

    M=cell2mat(M);
    RHS=cell2mat(RHS);
    C=cat(1, C{:});

    % Solve system
    [C]=solvesyst(C,M,RHS,Solver);

    % Reshape C to cell array
    C=mat2cell(C, n*ones(q,1), n);

    % Update coefficients beta and delta
    for i=1:q
        Va{i}=reshape((Am*C{i})', [n^2, r]);
        delta(:,i)=sum(Va{i}(1:(n+1):n^2,:),1);
    end

    for i=1:q
        for j=1:q
            beta(:,:,i,j)=Va{i}'*Va{j};
        end
    end

    % Compute residual
    residual=sqrt(n*m-2*sum(gamma.*delta, 'all')+sum(alpha.*beta, 'all'));
    res(niter)=residual;
    niter=niter+1;
end

% Convert output
[C]=convert(C, fout(1));
[D]=convert(D, fout(2));

varargout{1}=res;
end