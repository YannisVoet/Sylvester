% Experiment 4: RC circuit
% References:
% [1] Z. Bai and D. Skoogh. A projection method for model reduction of
% bilinear dynamical systems. Linear algebra and its applications, 2006.
% [2] P. Benner and T. Breiten. Low rank methods for a class of generalized
% Lyapunov equations and related issues. Numerische Mathematik, 2013.

clc
clear variables
close all

%% Set data
n0=30;
n=n0+n0^2;
m=n;
r=3;

e=ones(n0,1);
K1=spdiags([41*e -82*e 41*e], -1:1, n0, n0);
K1(n0,n0)=-41;

I=[1 1 1 1 2 2 2 2 2 2];
J=[1 2 n0+1 n0+2 1 2 n0+1 n0+3 2*n0+2 2*n0+3];
V=[-1600 800 800 -800 800 -800 -800 800 800 -800];

for k=3:n0-2
    I=[I k k k k k k];
    J=[J (k-2)*n0+k-1 (k-2)*n0+k (k-1)*n0+k-1 (k-1)*n0+k+1 k*n0+k k*n0+k+1];
    V=[V 800 -800 -800 800 800 -800];
end

I=[I n0 n0 n0 n0];
J=[J (n0-2)*n0+n0-1 (n0-2)*n0+n0 (n0-1)*n0+n0-1 (n0-1)*n0+n0];
V=[V 800 -800 -800 800];

K2=sparse(I,J,V,n0,n0^2);
Id=speye(n0,n0);
b=Id(:,1);

M=[K1 K2; sparse(n0^2,n0) kron(K1,Id)+kron(Id,K1)];
N=[sparse(n0,n0) sparse(n0,n0^2); kron(b,Id)+kron(Id,b) sparse(n0^2, n0^2)];
b=[b; sparse(n0^2,1)];

A=cell(1,r);
B=cell(1,r);

A{1}=speye(n);
A{2}=M;
A{3}=N;

B{1}=M;
B{2}=speye(n);
B{3}=N;

RHS=-b*b';
X0=sparse(m,n);

%% Define linear operator
linearOp = @(X) linOp(A,B,X,n,m);

%% Preconditioner using the Lyapunov part of the equation
tic
[P,S,Q1,Z1]=qz(full(B{1}),full(B{2}),'real');
[T,R,Q2,Z2]=qz(full(A{2}),full(A{1}),'real');
time_lyap.setup=toc;

prec_lyap = @(X) Z1*rtrgsyl(P,R,S,T,Q1*X*Q2')*Z2';

%% Nearest Kronecker product approximation
% q-values for NKP preconditioners
q_nkp=[1 2];
b_nkp=length(q_nkp);

prec_nkp=cell(1,b_nkp);
setup_fact=cell(1,b_nkp);

tic
[Y,Z] = kronsvd(A,B,min([2 max(q_nkp)]));
setup_nkp=toc;

for k=1:b_nkp
    switch q_nkp(k)
        case 1
            tic
            dY=decomposition(Y{1}');
            dZ=decomposition(Z{1});
            setup_fact{k}=toc;

            prec_nkp{k} = @(X) dZ\(X/dY);
        otherwise
            tic
            [P,S,Q1,Z1]=qz(Z{1},Z{2},'real');
            [T,R,Q2,Z2]=qz(Y{2},Y{1},'real');
            setup_fact{k}=toc;

            prec_nkp{k} = @(X) Z1*rtrgsyl(P,R,S,T,Q1*X*Q2')*Z2';
    end
end

time_nkp=struct('setup_nkp', setup_nkp, 'setup_fact', setup_fact);

%% Low Kronecker rank approximation of the inverse
% q-values for KINV preconditioners
q_kinv=[2 4];
b_kinv=length(q_kinv);

prec_kinv=cell(1,b_kinv);
res_kinv=cell(1,b_kinv);
setup_kinv=cell(1,b_kinv);

for k=1:b_kinv
    tic
    [Cin]=sparinv(A,1:q_kinv(k));
    [Din]=sparinv(B,1:q_kinv(k));
    [C, D, res_kinv{k}] = kroninvq(A, B, q_kinv(k), 'C0', Cin, 'D0', Din, 'nitermax', 10, 'sparse', true, 'parallel', true);
    setup_kinv{k}=toc;
    prec_kinv{k} = @(X) linOp(C,D,X,n,m);
end

time_kinv=struct('setup_kinv', setup_kinv);

%% GMRES Solver
norm_RHS=norm(RHS, 'fro');
maxiter=100;
tol=1e-8*norm_RHS; % Relative tolerance
restart=50;

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glgmresk(linearOp, RHS, restart, tol, maxiter, @(X) X, X0);
time_unprec.gmres=toc;
tic
[X_prec_lyap, res_prec_lyap, iter_lyap] = glgmresk(linearOp, RHS, restart, tol, maxiter, prec_lyap, X0);
time_lyap.prec_gmres=toc;

for k=1:b_nkp
    tic
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glgmresk(linearOp, RHS, restart, tol, maxiter, prec_nkp{k}, X0);
    time_nkp(k).prec_gmres=toc;
end

for k=1:b_kinv
    tic
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glgmresk(linearOp, RHS, restart, tol, maxiter, prec_kinv{k}, X0);
    time_kinv(k).prec_gmres=toc;
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};

figure
semilogy(res_unprec/norm_RHS, '-xk','DisplayName', 'GMRES')
hold on; grid on;
semilogy(res_prec_lyap/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Lyapunov part')

for k=1:b_nkp
    semilogy(res_prec_nkp{k}/norm_RHS, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP(' num2str(min([q_nkp(k) 2])) ')'])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}/norm_RHS, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV(' num2str(q_kinv(k)) ')'])
end


legend show
legend('Location','northeast')
xlabel('Iteration number')
ylabel('Relative residual')

%% Bi-CGSTAB Solver
norm_RHS=norm(RHS, 'fro');
maxiter=100;
tol=1e-8*norm_RHS; % Relative tolerance

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glbicgstb(linearOp, RHS, tol, maxiter, @(X) X, X0);
time_unprec.bicgstab=toc;
tic
[X_prec_lyap, res_prec_lyap, iter_lyap] = glbicgstb(linearOp, RHS, tol, maxiter, prec_lyap, X0);
time_lyap.prec_bicgstab=toc;

for k=1:b_nkp
    tic
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_nkp{k}, X0);
    time_nkp(k).prec_bicgstab=toc;
end

for k=1:b_kinv
    tic
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_kinv{k}, X0);
    time_kinv(k).prec_bicgstab=toc;
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};

figure
semilogy(res_unprec/norm_RHS, '-xk','DisplayName', 'Bi-CGSTAB')
hold on; grid on;
semilogy(res_prec_lyap/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Lyapunov part')

for k=1:b_nkp
    semilogy(res_prec_nkp{k}/norm_RHS, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP(' num2str(min([q_nkp(k) 2])) ')'])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}/norm_RHS, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV(' num2str(q_kinv(k)) ')'])
end


legend show
legend('Location','northeast')
xlabel('Iteration number')
ylabel('Relative residual')

