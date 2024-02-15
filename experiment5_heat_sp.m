% Experiment 5: Heat transfer model with singular Lyapunov part
% References:
% [1] P. Benner and T. Damm. Lyapunov equations, energy functionals, and 
% model order reduction of bilinear and stochastic systems. 
% SIAM journal on control and optimization, 2011.
% [2] P. Benner and T. Breiten. Low rank methods for a class of generalized 
% Lyapunov equations and related issues. Numerische Mathematik, 2013.
% [3] T. Damm. Direct methods and adi-preconditioned krylov subspace 
% methods for generalized lyapunov equations. Numerical Linear Algebra 
% with Applications, 2008.

clc
clear variables
close all

%% Set data
n0=15;
n=n0^2;
m=n;
r=2;

e=ones(n0,1);
Id=speye(n0,n0);
T=spdiags([e -2*e e], -1:1, n0, n0);

e1=Id(:,1);
en=Id(:,end);
E1=e1*e1';
En=en*en';

K=(n0+1)^2*(kron(Id,T+E1+En)+kron(T+E1+En,Id));
N1=(n0+1)*kron(E1,Id);
N2=(n0+1)*kron(Id,E1);
N3=(n0+1)*kron(En,Id);
N4=(n0+1)*kron(Id,En);

b=(n0+1)*[kron(e1,e) kron(e,e1) kron(en,e) kron(e,en)];

A=cell(1,r);
B=cell(1,r);

A{1}=speye(n);
A{2}=K;
A{3}=N1;
A{4}=N2;
A{5}=N3;
A{6}=N4;

B{1}=K;
B{2}=speye(n);
B{3}=N1;
B{4}=N2;
B{5}=N3;
B{6}=N4;

RHS=-b*b';
X0=sparse(m,n);

%% Define linear operator
linearOp = @(X) linOp(A,B,X,n,m);

%% Nearest Kronecker product approximation
% q-values for NKP preconditioners
q_nkp=[1 2];
b_nkp=length(q_nkp);

prec_nkp=cell(1,b_nkp);

[Y,Z] = kronsvd(A,B,min([2 max(q_nkp)]));

for k=1:b_nkp
    switch q_nkp(k)
        case 1
            prec_nkp{k} = @(X) Z{1}\(X/(Y{1}'));
        otherwise
            [P,S,Q1,Z1]=qz(Z{1},Z{2},'real');
            [T,R,Q2,Z2]=qz(Y{2},Y{1},'real');

            prec_nkp{k} = @(X) Z1*rtrgsyl(P,R,S,T,Q1*X*Q2')*Z2';
    end
end

%% Low Kronecker rank approximation of the inverse
% q-values for KINV preconditioners
q_kinv=[1 2 3 4];
b_kinv=length(q_kinv);

prec_kinv=cell(1,b_kinv);
res_kinv=cell(1,b_kinv);

for k=1:b_kinv
    [Cin]=sparinv(A,2+(1:q_kinv(k)));
    [Din]=sparinv(B,2+(1:q_kinv(k)));
    [C, D, res_kinv{k}] = kroninvq(A, B, q_kinv(k), 'C0', Cin, 'D0', Din, 'nitermax', 10, 'sparse', true);
    prec_kinv{k} = @(X) linOp(C,D,X,n,m);
end

%% GMRES Solver
norm_RHS=norm(RHS, 'fro');
maxiter=100;
tol=1e-8*norm_RHS; % Relative tolerance

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

[X_unprec, res_unprec, iter_unprec] = glgmresk(linearOp, RHS, [], tol, maxiter, @(X) X, X0);

for k=1:b_nkp
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_nkp{k}, X0);
end

for k=1:b_kinv
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_kinv{k}, X0);
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};

figure
semilogy(res_unprec/norm_RHS, '-xk','DisplayName', 'GMRES')
hold on; grid on;

for k=1:b_nkp
    semilogy(res_prec_nkp{k}/norm_RHS, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP prec, q=' num2str(min([q_nkp(k) 2]))])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}/norm_RHS, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV prec, q=' num2str(q_kinv(k))])
end


legend show
legend('Location','northeast')
xlabel('Iteration number')
ylabel('Relative residual')

%% Bi-CGSTAB Solver
maxiter=100;
tol=1e-8;

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

[X_unprec, res_unprec, iter_unprec] = glbicgstb(linearOp, RHS, tol, maxiter, @(X) X, X0);

for k=1:b_nkp
    [X_prec_nkp, res_prec_nkp{k}, iter_prec_nkp{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_nkp{k}, X0);
end

for k=1:b_kinv
    [X_prec_kinv, res_prec_kinv{k}, iter_prec_kinv{k}] = glbicgstb(linearOp, RHS, tol, maxiter, prec_kinv{k}, X0);
end

%% Results

colors_nkp={[0 0.8 1], [0.9 0.5 0.1]};
colors_kinv={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1],[0.8 0.8 0.2],[0.8 0.4 0.8]};

linespec_nkp={'v', '^'};
linespec_kinv={'s', '*', 'o', 'square', 'diamond'};

figure
semilogy(res_unprec, '-xk','DisplayName', 'Bi-CGSTAB')
hold on; grid on;

for k=1:b_nkp
    semilogy(res_prec_nkp{k}, 'Marker', linespec_nkp{q_nkp(k)}, 'Color', colors_nkp{q_nkp(k)}, 'DisplayName', ['NKP prec, q=' num2str(min([q_nkp(k) 2]))])
end

for k=1:b_kinv
    semilogy(res_prec_kinv{k}, 'Marker', linespec_kinv{q_kinv(k)}, 'Color', colors_kinv{q_kinv(k)}, 'DisplayName', ['KINV prec, q=' num2str(q_kinv(k))])
end


legend show
legend('Location','northeast')
xlabel('Iteration number')
ylabel('Residual')
