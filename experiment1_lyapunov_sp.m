% Experiment 1: standard Lyapunov equation
% Validation experiment

clc
clear variables
close all

%% Set data
n=200;
m=n;
r=2;
e=ones(n,1);
K=(n+1)^2*spdiags([-e 2*e -e], -1:1, n, n);

A=cell(1,r);
B=cell(1,r);

A{1}=speye(n);
A{2}=K;

B{1}=K;
B{2}=speye(n);

RHS=ones(m,n);
X0=zeros(m,n);

%% Define linear operator
linearOp = @(X) linOp(A,B,X,n,m);

%% Nearest Kronecker product approximation
q=1;

[Y,Z,err] = kronsvd(A,B,q);

prec_nkp= @(X) Z{1}\(X/(Y{1}'));

%% Low Kronecker rank approximation of the inverse
b=3;

prec_kinvq=cell(1,b);
res_kinvq=cell(1,b);

for k=1:b

    Cin=cell(1,k);
    Din=cell(1,k);

    switch k

        case 1 % Kronecker rank 1 approximate inverse
            b1=3;
            Cin{1}=spdiags(ones(n, 2*b1+1), -b1:b1, n, n);
            Din{1}=spdiags(ones(m, 2*b1+1), -b1:b1, m, m);

        case 2 % Kronecker rank 2 approximate inverse
            b1=10;
            b2=b1+1;
            Cin{1}=spdiags(ones(n, 2*b1+1), -b1:b1, n, n);
            Cin{2}=spdiags(ones(n, 2*b2+1), -b2:b2, n, n);
            Din{1}=spdiags(ones(m, 2*b1+1), -b1:b1, m, m);
            Din{2}=spdiags(ones(m, 2*b2+1), -b2:b2, m, m);

        case 3 % Kronecker rank 3 approximate inverse
            b1=20;
            b2=b1+1;
            b3=b2+1;
            Cin{1}=spdiags(ones(n, 2*b1+1), -b1:b1, n, n);
            Cin{2}=spdiags(ones(n, 2*b2+1), -b2:b2, n, n);
            Cin{3}=spdiags(ones(n, 2*b3+1), -b3:b3, n, n);
            Din{1}=spdiags(ones(m, 2*b1+1), -b1:b1, m, m);
            Din{2}=spdiags(ones(m, 2*b2+1), -b2:b2, m, m);
            Din{3}=spdiags(ones(m, 2*b3+1), -b3:b3, m, m);

    end

    [C, D, res_kinvq{k}] = kroninvq(A, B, k, 'C0', Cin, 'D0', Din, 'sparse', true);
    prec_kinvq{k} = @(X) linOp(C,D,X,n,m);

end


%% Solver
maxiter=100;
tol=1e-8;

[X_unprec, res_unprec, iter_unprec] = glgmresk(linearOp, RHS, [], tol, maxiter, @(X) X, X0);
[X_prec_nkp, res_prec_nkp, iter_prec_nkp] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_nkp, X0);

res_prec_kinvq=cell(1,b);
inter_prec_kinv=cell(1,b);

for k=1:b
    [~, res_prec_kinvq{k}, inter_prec_kinv{k}] = glgmresk(linearOp, RHS, [], tol, maxiter, prec_kinvq{k}, X0);
end

%% Results

colors_nkp={[0 0.8 1]};
colors_kinvq={[0 0.2 0.8],[0.9 0.1 0.1],[0.4 0.7 0.1]};

linespec_nkp={'v'};
linespec_kinvq={'s', '*', 'o'};

figure
semilogy(res_unprec, '-xk','DisplayName', 'GMRES')
hold on; grid on;
semilogy(res_prec_nkp, 'Marker', linespec_nkp{1}, 'Color', colors_nkp{1}, 'DisplayName', 'NKP(1)')

for k=1:b
    semilogy(res_prec_kinvq{k}, 'Marker', linespec_kinvq{k}, 'Color', colors_kinvq{k}, 'DisplayName', ['KINV(' num2str(k) ')'])
end

legend show
legend('Location','southeast')
xlabel('Iteration number')
ylabel('Residual')

