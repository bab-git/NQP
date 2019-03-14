function [X,fobj]=NQP(H,C,T0)
% The non-negative quadratic pursuit algorithm
%
% This code solves the following problem:
%
%    minimize_x 0.5*x'*H*x + C'*x
%         s.t   x>=0  , ||x||_0<T0
%
% The detail of the algorithm is described in the following paper:
% 'Confident kernel dictionary learning for discriminative representation
%  of multivariate time-series', B. Hosseini, F. Petitjean, Forestier G., and B. Hammer.
%
% Written by Babak Hosseini <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
% Copyright 2019 by Babak Hosseini
% %
if size(C,1)<size(C,2)
    C=C';
end
e=(eig(H));
if min(e)<0 && abs(min(e)/max(e))>1e-4
    error('H is indifinite!')
end
if norm(H-H')>1e-4
    error('H is not symmetric!')
end

e_tol=1e-6;
e_conv=1e-3;
N = size(H,1);
X = sparse(zeros(N,1));
S = []; % positions indexes of components of s
R = 1:N; % positions indexes of components of s
R(sum(H,1)==0)=[];
res_phi = C; % first r*d

res_x=100;
x_est0=zeros(N,1);
x_est=[];
t=1;
while (t<=T0)
    if isempty(find(res_phi(R)<-e_tol))
        break
    end
    
    [~,j]=max(-res_phi(R));  % most negative gradiant
    j=R(j);
    S = [S j];
    R=R(R~=j);
    Hi=H(S,S);
    Ci=C(S);
    fx=@(x) 0.5*x'*Hi*x+Ci'*x;
    %%====nn QP
    if t==1
        L=sqrt(Hi);
    else
        v=H(S(S~=j),j);
        w=L_1\v;
        c=H(j,j);
        if (c-w'*w) <1e-4
            %             disp('dependant!!')
            S(S==j)=[];
            continue
        end
        L=[L_1 zeros(size(L_1,1),1) ; w' sqrt(c-w'*w)];
    end
    x_est=  -(L')\(L\Ci);
    
    if sum((sign(x_est) < 0))  % first zero crossing
        neg_flag=1;
        x20=zeros(size(H,1),1); x20(S0)=x_est0;
        x20=x20(S);
        Hi2=Hi;
        Ci2=Ci;
        while (1)
            if isempty(S)
                error('empty(S)')
            end
            progress = (0 - x20)./(x_est - x20);
            i_n=find(x_est<0);
            p_neg=progress(i_n);
            [v, i_p]=min(p_neg);
            
            remov=i_n(i_p);
            Hi2(remov,:)=[];
            Hi2(:,remov)=[];
            Ci2(remov)=[];
            S(remov)=[];
            L=chol(Hi2,'lower');
            x_est=  -(L')\(L\Ci2);
            
            fx=@(x) 0.5*x'*Hi2*x+Ci2'*x;
            x_est(x_est<1e-5)=0;
            if sum((sign(x_est) < 0))
                x20(remov)=[];
            else
                break
            end
        end
        
    else
        neg_flag=0;
    end
    
    res_x0=fx(x_est);
    
    if norm(res_x0-res_x)/norm(res_x) > e_tol && ~neg_flag % positive progress
        res_x=res_x0;
    elseif neg_flag==0  % no progress
        S=S(S~=j);
        if isempty(S)
            x_est=0;
        else
            L=L_1;
            x_est=x_est0;
        end
    else % zero-crossing case
        res_x=res_x0; % compare next round to current zero-crossing x
    end
    
    if isempty(S)
        res_phi=C';
    else
        res_phi=C'+x_est'*H(S,:);
    end
    S0=S;
    x_est0=x_est;
    
    L_1=L;
    if sum(x_est) >0
        t=numel(x_est)+1;
    end
    if abs(res_x) < e_conv
        disp('The minimum objective threshold is reached')
        disp('Change e_conv if you want a smaller objective value')
    end
    if isempty(R)%|| norm(res_phi) < e_tol
        break
    end
    
    
end
if isempty(x_est)
    disp('No valid solution, please check the validity of H and C')
    X=zeros(N,1);
else
    X(S)=x_est;
    disp('Optimization is Convereged')
end
X(X/max(X)<1e-2)=0;
fx=@(x) 0.5*x'*H*x+C'*x;
fobj=fx(X);