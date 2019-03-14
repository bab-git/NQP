% This is the demo use of the Non-negatice Quadratic Pursuit algorithm
% Written by Babak Hosseini <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
% Copyright 2019 by Babak Hosseini

clear;clc

n=100;
T0=20;
lam=1;
H=rand(n);
H(randperm(n,floor(n/4)),:)=0;
H=H*H';
C=rand(n,1)-5;
C(randperm(n,floor(n/4)))=0;
[x,f]=NQP(H,C,T0)
Objective_value=@(x) 0.5*x'*H*x+C'*x;