function[X, varargout] = glbicgstb(A, B, varargin)

% GLBICGSTB: Global Bi-CGSTAB algorithm for solving linear matrix equations
% A(X) = B where X and B are m x n matrices and A is a linear operator.
% Based on Algorithm 7.7 of [1].
%
% X = GLBICGSTB(A, B) iteratively solves the matrix equation A(X) = B.
%
% X = GLBICGSTB(A, B, tol) specifies the tolerance on the absolute residual. 
% For a stopping criterion based on the relative residual, replace tol 
% with tol*norm(b). Default: 1e-8.
%
% X = GLBICGSTB(A, B, tol, maxiter) also specifies the maximum number
% of iterations. Default: min(mn,10).
%
% X = GLBICGSTB(A, b, tol, maxiter, M) also specifies a preconditioning
% operator. The algorithm uses right preconditioning.
%
% X = GLBICGSTB(A, B, tol, maxiter, M, X0) also specifies the initial
% starting matrix. Default: all zero matrix.
%
% [X, res] = GLBICGSTB(A, B, ...) returns the vector of absolute residuals 
% for each iteration.
%
% [X, res, niter] = GLBICGSTB(A, B, ...) also returns the number of iterations.
% Non-integer values of niter indicate convergence halfway through an
% iteration.
%
% Reference:
% [1] Y. Saad. Iterative methods for sparse linear systems. SIAM, 2003.
% [2] H. A. Van der Vorst. Bi-CGSTAB: A fast and smoothly converging 
% variant of Bi-CG for the solution of nonsymmetric linear systems. 
% SIAM Journal on scientific and Statistical Computing, 1992.
%
% See also GLGMRESK, GLCG.

%% Set algorithm parameters

% Set default parameters
Default{1}=1e-8;
Default{2}=min(numel(B),10);
Default{3}=@(X) X;
Default{4}=sparse(size(B));

% Replace empty inputs with default parameters
def=cell2mat(cellfun(@isempty, varargin, 'UniformOutput', false));
[varargin{def}]=Default{def};

Param = inputParser;
Param.addRequired('A', @(x) isa(x, 'function_handle') || isa(x, 'double'));
Param.addRequired('B');
Param.addOptional('tol', Default{1}, @(x) isscalar(x) && x > 0);
Param.addOptional('maxiter', Default{2}, @(x) isscalar(x) && x > 0);
Param.addOptional('M', Default{3}, @(x) isa(x, 'function_handle') || isa(x, 'double'));
Param.addOptional('X0', Default{4}, @(x) all(size(x)==size(B)));
Param.parse(A, B, varargin{:});

%% Retrieve parameters
tol=Param.Results.tol;
maxiter=Param.Results.maxiter;
M=Param.Results.M;
X0=Param.Results.X0;

%% Bi-CGSTAB method

R0=B-A(X0);
R=R0;
P=R0;
X=X0;
Rh=R0; % MATLAB's choice
% Rh=rand(size(X0));

res=zeros(2*maxiter+1,1);
res(1)=norm(R0, 'fro');
rhoo=sum(conj(Rh).*R, 'all');
hit=1;

for k=1:maxiter

    V=M(P);
    V=A(V);
    alpha=rhoo/sum(conj(Rh).*V, 'all');
    X=X+alpha*P;
    S=R-alpha*V;
    res(2*k)=norm(S, 'fro');

    if res(2*k)<tol
        hit=0;
        break
    end

    T=M(S);
    T=A(T);
    omega=sum(conj(T).*S, 'all')/(norm(T, 'fro')^2);
    X=X+omega*S;
    R=S-omega*T;
    res(2*k+1)=norm(R, 'fro');

    if res(2*k+1)<tol
        break
    end

    rhon=sum(conj(Rh).*R, 'all');
    beta=(rhon/rhoo)*(alpha/omega);
    P=R+beta*(P-omega*V);
    rhoo=rhon;

end

X=M(X);
varargout{1}=res(1:2*k+hit);
varargout{2}=k+0.5*(hit-1);

end