function[varargout]=aca(A, varargin)

% ACA: Adaptive cross approximation with full pivoting.
%
% [U,V] = ACA(A) computes a low rank approximation A ≈ U*V.
%
% [U,V] = ACA(A, tol) specifies the tolerance. Default: 1e-9.
%
% [U,V] = ACA(A, tol, rank) also specifies the maximum rank.
% Default: min(size(A), 30).
%
% [U,V,I,J] = ACA(A, ...) returns the indices of the rows and columns
% selected.
%
% [U,V,I,J,k] = ACA(A, ...) also returns the rank of the approximation.
%
% [U,V,I,J,k,err] = ACA(A, ...) also returns the relative approximation
% error.
%
% See also ACAP.

%% Set algorithm parameters

Param = inputParser;
Param.addRequired('matrix');
Param.addOptional('tolerance', 1e-9, @(x) isscalar(x) & x > 0);
Param.addOptional('rank', min([size(A), 30]), @(x) isscalar(x) & x > 0);
Param.parse(A, varargin{:});

%% Retrieve parameters
tol=Param.Results.tolerance;
l=Param.Results.rank;

%% Adaptive cross approximation
[m,n]=size(A);
norm_A=norm(A);
R=A;

% Initialization
I=zeros(l,1); J=zeros(l,1);
U=zeros(m,l); V=zeros(n,l);
err=zeros(l,1);

for k=1:l
    [~, indx]=max(abs(R), [], 'all');
    [I(k),J(k)]=ind2sub([m,n],indx);
    delta=R(I(k), J(k));
    U(:,k)=R(:,J(k));
    V(:,k)=1/delta*R(I(k),:)';
    R=R-U(:,k)*V(:,k)';

    if nargout>5
        err(k)=norm(R)/norm_A;
    end

    if norm(U(:,k))*norm(V(:,k)) <= tol*norm(U(:,1:k)*V(:,1:k)', 'fro')
        break
    end
end

if norm(U(:,k))*norm(V(:,k)) > tol*norm(U(:,1:k)*V(:,1:k)', 'fro')
    fprintf('The adaptive cross approximation did not convergence to the desired tolerance for the prescribed maximum rank \n')
end

varargout{1}=U(:,1:k);
varargout{2}=V(:,1:k);
varargout{3}=I(1:k);
varargout{4}=J(1:k);
varargout{5}=k;
varargout{6}=err(1:k);
end