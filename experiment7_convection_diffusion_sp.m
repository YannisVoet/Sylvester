% Experiment 7: Convection-diffusion equation
% Reference:
% [1] D. Palitta and V. Simoncini. Matrix-equation-based strategies for 
% convection–diffusion equations. BIT Numerical Mathematics, 2016.

clc
clear variables
close all

%% Set data
n=1000; % Number of nodes
m=n;
r=3;
N=n-1; % Number of subdivisions

% Diffusion coefficient
epsilon=1/10;

e=ones(n,1);

% Grid nodes
x=linspace(0,1,m);
y=linspace(0,1,n);

f=@(x,y) zeros(size(x))+zeros(size(y));

B1=spdiags([-e 0*e e], -1:1, n, n);
B1(1,1)=0;  B1(end,end)=0;
B1(1,2)=0;  B1(end,end-1)=0;

B1=0.5*N*B1;
B2=B1';

% Convection coefficients
phi1 = @(x) 1-(2*x+1).^2;
psi1 = @(y) y;

phi2 = @(x) -2*(2*x+1);
psi2 = @(y) 1-y.^2;

Phi1=diag(phi1(x));
Psi1=diag(psi1(y));

Phi2=diag(phi2(x));
Psi2=diag(psi2(y));

T1=spdiags([-e 2*e -e], -1:1, n, n);

T1(1,1)=1; T1(end,end)=1;
T1(1,2)=0; T1(end,end-1)=0;

T1=epsilon*N^2*T1;
T2=T1';

A=cell(1,r);
B=cell(1,r);

A{1}=speye(n);
A{2}=T1;
A{3}=Psi1;
A{4}=Psi2*B1;

B{1}=T1;
B{2}=speye(m);
B{3}=Phi1*B1;
B{4}=Phi2;

%% Right-hand side
% Boundary conditions
fx0 = @(y) zeros(size(y));
fx1 = @(y) zeros(size(y));

fy0 = @(x) (1+tanh(10+20*(2*x-1))).*(0 <= x & x <= 0.5) + 2*(0.5 < x & x <= 1);
fy1 = @(x) zeros(size(x));

RHS=f(x',y);

% Side x=0
RHS(1,:)=epsilon*N^2*fx0(y)+fx0(y)*T2+phi2(x(1))*fx0(y)*B2*Psi2;
% Side x=1
RHS(end,:)=epsilon*N^2*fx1(y)+fx1(y)*T2+phi2(x(end))*fx1(y)*B2*Psi2;
% Side y=0
RHS(:,1)=T1*fy0(x')+epsilon*N^2*fy0(x')+psi1(y(1))*Phi1*B1*fy0(x');
% Side y=1
RHS(:,end)=T1*fy1(x')+epsilon*N^2*fy1(x')+psi1(y(end))*Phi1*B1*fy1(x');

X0=sparse(m,n);

VA=cell2vect(A);
VB=cell2vect(B);

GA=VA'*VA;
GB=VB'*VB;

dA=1./sqrt(diag(GA));
dB=1./sqrt(diag(GB));

CA=full(dA.*GA.*dA');
CB=full(dB.*GB.*dB');

%% Define linear operator
linearOp = @(X) linOp(A,B,X,n,m);

%% Preconditioner of Palitta and Simoncini, 2016

ps1=mean(diag(Psi1));
ph2=mean(diag(Phi2));

G{1}=speye(n);
G{2}=(T2+ph2*B2*Psi2)';

H{1}=T1+ps1*Phi1*B1;
H{2}=speye(n);

tic
[P,S,Q1,Z1]=qz(full(H{1}),full(H{2}),'real');
[T,R,Q2,Z2]=qz(full(G{2}),full(G{1}),'real');
time_sylv.setup=toc;

prec_sylv = @(X) Z1*rtrgsyl(P,R,S,T,Q1*X*Q2')*Z2';

%% Nearest Kronecker product approximation
% q-values for NKP preconditioners
q_nkp=[1 2];
b_nkp=length(q_nkp);

prec_nkp=cell(1,b_nkp);
setup_fact=cell(1,b_nkp);

tic
[Y,Z,err] = kronsvd(A,B,min([2 max(q_nkp)]));
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
    [Cin]=sparinv(A,15+(1:q_kinv(k)),[], @(x) abs(x)'*abs(x));
    [Din]=sparinv(B,15+(1:q_kinv(k)),[], @(x) abs(x)'*abs(x));
    [C, D, res_kinv{k}] = kroninvq(A, B, q_kinv(k), 'C0', Cin, 'D0', Din, 'nitermax', 5, 'sparse', true, 'parallel', true);
    setup_kinv{k}=toc;
    prec_kinv{k} = @(X) linOp(C,D,X,n,m);
end

time_kinv=struct('setup_kinv', setup_kinv);

%% GMRES Solver
norm_RHS=norm(RHS, 'fro');
maxiter=200;
tol=1e-6*norm_RHS; % Relative tolerance
restart=[];

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glgmresk(linearOp, RHS, restart, tol, maxiter, @(X) X, X0);
time_unprec.gmres=toc;
tic
[X_prec_sylv, res_prec_sylv, iter_sylv] = glgmresk(linearOp, RHS, restart, tol, maxiter, prec_sylv, X0);
time_sylv.prec_gmres=toc;

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
semilogy(res_prec_sylv/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Palitta and Simoncini, 2016')

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
maxiter=200;
tol=1e-6*norm_RHS; % Relative tolerance

res_prec_nkp=cell(1,b_nkp);
res_prec_kinv=cell(1,b_kinv);

iter_prec_nkp=cell(1,b_nkp);
iter_prec_kinv=cell(1,b_kinv);

tic
[X_unprec, res_unprec, iter_unprec] = glbicgstb(linearOp, RHS, tol, maxiter, @(X) X, X0);
time_unprec.bicgstab=toc;
tic
[X_prec_sylv, res_prec_sylv, iter_sylv] = glbicgstb(linearOp, RHS, tol, maxiter, prec_sylv, X0);
time_sylv.prec_bicgstab=toc;

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
semilogy(res_prec_sylv/norm_RHS, 'Marker', 'pentagram', 'Color', [0.4660 0.6740 0.1880], 'DisplayName', 'Palitta and Simoncini, 2016')

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
