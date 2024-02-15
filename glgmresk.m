function[X, varargout] = glgmresk(A, B, varargin)

% GLGMRESK: (Restarted) Global Generalized Minimal Residual Method
% (Gl-GMRES(k)) for solving linear matrix equations A(X) = B where X and B
% are m x n matrices and A is a linear operator.
%
% X = GLGMRESK(A, B) iteratively solves the matrix equation A(X) = B.
%
% X = GLGMRESK(A, B, restart) restarts the method every restart iterations.
% If this parameter is mn or [] the unrestarted method is used.
% Default: mn (unrestarted method).
%
% X = GLGMRESK(A, B, restart, tol) specifies the tolerance on the absolute 
% residual. For a stopping criterion based on the relative residual, 
% replace tol with tol*norm(b). Default: 1e-8.
%
% X = GLGMRESK(A, B, restart, tol, maxiter) also specifies the maximum
% number of outer iterations (i.e. the maximum number of restarts). If the
% method is not restarted, the maximum number of total iterations is
% maxiter (instead of maxiter*restart).
% Default: min(mn,10).
%
% X = GLGMRESK(A, B, restart, tol, maxiter, M) also specifies a
% preconditioning operator. The algorithm uses the right preconditioned
% version of Gl-GMRES.
%
% X = GLGMRESK(A, B, restart, tol, maxiter, M, X0) also specifies the
% initial starting matrix. Default: all zero matrix.
%
% X = GLGMRESK(A, B, restart, tol, maxiter, M, X0, reorth_tol) also
% specifies the reorthogonalization tolerance for the modified Gram-Schmidt
% process. Default: 0.7.
%
% [X, res] = GLGMRESK(A, B, ...) returns the vector of absolute residuals
% for each iteration.
%
% [X, res, niter] = GLGMRESK(A, B, ...) also returns a vector with the
% number of outer, inner and total number of iterations.
% 1 ≤ niter(1) ≤ maxiter
% 1 ≤ niter(2) ≤ restart
% 1 ≤ niter(3) ≤ maxiter*restart (or maxiter for the unrestarted method)
%
% References:
% [1] K. Jbilou, A. Messaoudi, and H. Sadok. Global FOM and GMRES algorithms
% for matrix equations. Applied Numerical Mathematics, 1999.
% [2] Y. Saad. Iterative methods for sparse linear systems. SIAM, 2003.
%
% See also GLBICGSTB, GLCG.

%% Set algorithm parameters

% Set default parameters
Default{1}=numel(B);
Default{2}=1e-8;
Default{3}=min(numel(B),10);
Default{4}=@(X) X;
Default{5}=sparse(size(B));
Default{6}=0.7;

% Replace empty inputs with default parameters
def=cell2mat(cellfun(@isempty, varargin, 'UniformOutput', false));
[varargin{def}]=Default{def};

Param = inputParser;
Param.addRequired('A', @(x) isa(x, 'function_handle'));
Param.addRequired('B');
Param.addOptional('restart',    Default{1}, @(x) isscalar(x) & x > 0);
Param.addOptional('tol',        Default{2}, @(x) isscalar(x) & x > 0);
Param.addOptional('maxiter',    Default{3}, @(x) isscalar(x) & x > 0);
Param.addOptional('M',          Default{4}, @(x) isa(x, 'function_handle'));
Param.addOptional('X0',         Default{5}, @(x) all(size(x)==size(B)));
Param.addOptional('reorth_tol', Default{6}, @(x) isscalar(x) & x > 0);
Param.parse(A, B, varargin{:});

%% Retrieve parameters
restart=Param.Results.restart;
tol=Param.Results.tol;
maxiter=Param.Results.maxiter;
M=Param.Results.M;
X0=Param.Results.X0;
reorth_tol=Param.Results.reorth_tol;

if restart==numel(B)
    restart=maxiter;
    maxiter=1;
end

%% Global GMRES method

res = cell(maxiter,1);

for outerit=1:maxiter

    [X,beta,innerit]=innergmres(A,M,B,X0,restart,tol,reorth_tol);
    res{outerit}=beta;

    if beta(end)<tol
        break
    end
    X0 = X;
end

varargout{1}=sort(uniquetol(cat(1,res{:}),1e-14), 'descend');
varargout{2}=[outerit innerit (outerit-1)*restart+innerit];


    function[X,beta,k]=innergmres(A,M,B,X0,s,tol,reorth_tol)

        R0=B-A(X0);
        [n,m]=size(R0);
        U=zeros(n*m, s);
        H=zeros(s+1, s);
        beta=zeros(s+1,1);
        r0=R0(:);
        beta(1)=norm(r0);
        U(:,1)=r0/beta(1);

        for k=1:s
            W=M(reshape(U(:,k), [n m]));
            W=A(W);

            w=W(:);
            h=U(:,1:k)'*w;
            u_tilde=w-U(:,1:k)*h;

            if norm(u_tilde-w)<reorth_tol % Re-orthogonalization
                h_hat=U(:,1:k)'*u_tilde;
                h=h+h_hat;
                u_tilde=u_tilde-U(:,1:k)*h_hat;
            end

            h_tilde=norm(u_tilde);
            U(:,k+1)=u_tilde/h_tilde;
            H(1:(k+1) ,k)=[h; h_tilde];

            % Be careful: parentheses are crucial
            y=beta(1)*(H(1:k+1,1:k)\eye(k+1,1));
            beta(k+1)=norm(beta(1)*eye(k+1,1)-H(1:k+1,1:k)*y);

            if beta(k+1)<tol
                break
            end

        end
        V=M(reshape(U(:,1:k)*y, [n m]));
        X=X0+V;
        beta=beta(1:k+1);

    end

end