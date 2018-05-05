% @Author: hxy
% @Date:   2016-10-20 09:41:03
% @Last Modified by:   hxy
% @Last Modified time: 2018-05-05 11:18:56
function [best,record] = DECMSA(N,maxFEs,lb,ub,dim,fname,fno)
record = [];
p = 0.5;
c = 0.1;
KMean = 0.5;
FMean = 0.5;
CrMean = 0.5;
% max try time of generating a normal random variable
maxTry = 20;

X = init( rand(N,dim), lb, ub );
fitness = feval(fname,X',fno)';
A = [];
best = inf;
Xr2 = zeros(N,dim);
Cr = zeros(N,1);
F = zeros(N,1);
K = zeros(N,1);
FEs = N;
% initial variables
xmean = sum(X)/N;
C = cov(X);
[B,D] = eig(C);
% memory counter
mc = 1;

%% ...............................main loop...........................
enableLS = 1;
for iter = 1 : (maxFEs / N)
    [best,bestIdx] = min(fitness);
    record = [record;[best,FEs]];
    if enableLS > 0 && FEs/maxFEs > 0.90
        if max(diag(D)) > (ub-lb)/2
            break;
        else
            enableLS = 0;
        end
    end
  	K = KMean + randn(N,1)*0.3; 
   	F = FMean + randn(N,1)*0.2; 
    Cr = CrMean + randn(N,1)*0.1;
    for i = 1 : N
        % generate parameters according to random memory
        tryCount = 0;
        while F(i) <= 0 && tryCount < maxTry
            F(i) = FMean + 0.2*randn;
            tryCount = tryCount + 1;
        end
        if tryCount == maxTry
            F(i) = FMean + 0.1*randn;
        end
        K(i) = -1;
        tryCount = 0;
        while K(i) <= 0 && tryCount < maxTry
            K(i) = KMean + 0.3*randn;
            tryCount = tryCount + 1;
        end
        if tryCount == maxTry
            K(i) = KMean + 0.1*randn;
        end
    end
    Cr(Cr>1)=1;
    Cr(Cr<0)=0;
    F(F>1)=1;
    F(F<0)=0;
    K(K>1)=1;
    K(K<0)=0;
    %% generate random index of the first individuals used in mutation
    % r1 ~= i from X
    r1 = mod((1:N)'+randi(N-1,N,1),N);
    r1(r1==0)=N;
    % r2 ~= r1 ~= i from X or A
    r2 = randi(N+size(A,1),N,1);
    for i = 1 : N
        while r2(i) == i || r2(i) == r1(i)
            r2(i) = randi(N+size(A,1));
        end
    end
    Xr2(r2<=N,:) = X(r2(r2<=N),:);
    if any(r2>N)
        Xr2(r2>N,:) = A(r2(r2>N)-N,:);
    end
    %% generate Xbetter through sampling
    Xbettter = bsxfun(@plus,randn(N,dim)*sqrt(D)*B',xmean);
    V = X + bsxfun(@times,K,Xbettter-X) ...
        + bsxfun(@times,F,X(r1,:)-Xr2);
    
    %% binomial recombination
    U = X;
    rflag = bsxfun(@lt,rand(N,dim),Cr)';
    ttt = randi(dim,1,N) + (0:(N-1))*dim;
    rflag(ttt) = 1;
    rflag = rflag';
    U(rflag) = V(rflag);
    U(U<lb)=(lb+X(U<lb))/2;
    U(U>ub)=(ub+X(U>ub))/2;
    %% selection
    UFitness = feval(fname,U',fno)';
    improvement = UFitness - fitness;
    flag = UFitness <= fitness;
    FEs = FEs + N;
    
    %% add worse parents to archive
    % Remove duplicate elements
    tmpPop = unique([A;X(flag,:)],'rows');
    if size(tmpPop,1)<= N
        A = tmpPop;
    else
        % random remove something to make |A| <= N
        A = tmpPop(randperm(size(tmpPop,1),N),:);
    end
    
    %% update population
    X(flag,:) = U(flag,:);
    flag = improvement<0;
    fitness(flag) = UFitness(flag);
    
    %% update parameters
    if any(flag)
		[~,idx] = sort(-improvement(flag));
		w = zeros(1,sum(flag));
		w(idx) = ( 1 : length(w) ) .^ -0.5;
        w = w./sum(w);
       	KMean = (1-c)*KMean + c*w*K(flag).^2 / (w*K(flag));	 
       	FMean = (1-c)*FMean + c*w*F(flag);
        CrMean = (1-c)*CrMean + c*w*Cr(flag).^2 / (w*Cr(flag));  
    end
    
    %% update distribution
    xold = xmean;
    [~,sortedIdx] = sort(fitness);
    
M = ceil(p*N);
M(M>N)=N;
M(M<3)=3;
weights = log(M+1/2)-log(1:M)';
weights = weights/sum(weights);
mueff=sum(weights)^2/sum(weights.^2);
cc = (4+mueff/dim) / (dim+4 + 2*mueff/dim);
c1 = 0;
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((dim+2)^2+mueff));
    % select top p*N individuals
    selectedX = X(sortedIdx(1:M),:);
    
    % update mean value
    xmean = sum(bsxfun(@times,selectedX,weights));
    % calculate the full rank estimation of covariance matrix
    artmp = bsxfun(@minus,selectedX,xold);
    C = (1-cmu)*C + cmu*artmp'*diag(weights)*artmp;
    C = triu(C) + triu(C,1)';
    [B,D] = eig(C);
end
if FEs < maxFEs
    options = optimset('LargeScale','off',...
        'MaxFunEvals',maxFEs-FEs,...
        'Display','off');
    H = diag(inv(C));
    if ~any(isnan(H)) && ~any(isinf(H))
        options = optimset(options,...
            'InitialHessType','user-supplied',...
            'InitialHessMatrix',H );
    end
    [minfval,best] = fminunc(@(X) feval(fname,X',fno)',X(bestIdx,:),options);
    record = [record;[best,FEs]];
end