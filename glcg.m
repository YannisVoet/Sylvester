function[X, varargout] = glcg(A, B, varargin)

% GLCG: Global Conjugate Gradient algorithm for solving linear matrix 
% equations A(X) = B where X and B are m x n matrices and A is a linear 
% operator. Based on Algorithm 6.18 of [1].
%
% X = GLCG(A, B) iteratively solves the matrix equation A(X) = B.
%
% X = GLCG(A, B, tol) specifies the tolerance on the absolute residual. 
% For a stopping criterion based on the relative residual, replace tol 
% with tol*norm(b). Default: 1e-8.
%
% X = GLCG(A, B, tol, maxiter) also specifies the maximum number
% of iterations. Default: min(mn,10).
%
% X = GLCG(A, b, tol, maxiter, M) also specifies a preconditioning
% operator.
%
% X = GLCG(A, B, tol, maxiter, M, X0) also specifies the initial
% starting matrix. Default: all zero matrix.
%
% [X, res] = GLCG(A, B, ...) returns the vector of absolute residuals for
% each iteration.
%
% [X, res, niter] = GLCG(A, B, ...) also returns the number of iterations.
%
% Reference:
% [1] Y. Saad. Iterative methods for sparse linear systems. SIAM, 2003.
%
% See also GLGMRESK, GLBICGSTB.

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
Param.addOptional('tol',        Default{1}, @(x) isscalar(x) && x > 0);
Param.addOptional('maxiter',    Default{2}, @(x) isscalar(x) && x > 0);
Param.addOptional('M',          Default{3}, @(x) isa(x, 'function_handle') || isa(x, 'double'));
Param.addOptional('X0',         Default{4}, @(x) all(size(x)==size(B)));
Param.parse(A, B, varargin{:});

%% Retrieve parameters
tol=Param.Results.tol;
maxiter=Param.Results.maxiter;
M=Param.Results.M;
X=Param.Results.X0;

if isa(A, 'double')
    A=@(x) A*x;
end

if isa(M, 'double')
    dM=decomposition(M);
    M=@(x) dM\x;
end

%% CG method

R=B-A(X);
Z=M(R);
P=Z;
Q=A(P);
xi=sum(conj(P).*Q, 'all');
eta=sum(conj(R).*Z, 'all');

res=zeros(maxiter+1,1);
res(1)=norm(R, 'fro');


for k=1:maxiter

    alpha=eta/xi;
    X=X+alpha*P;
    R=R-alpha*Q;
    Z=M(R);
    eta=sum(conj(R).*Z, 'all');
    beta=eta/(alpha*xi);
    P=Z+beta*P;
    Q=A(P);
    xi=sum(conj(P).*Q, 'all');
    res(k+1)=norm(R, 'fro');

    if res(k+1)<tol
        break
    end

end

varargout{1}=res(1:k+1);
varargout{2}=k;

end