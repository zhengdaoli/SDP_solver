function [X,y,Z,out] = sdp_solver(A,b,C,tol)


%% check shape, sparsity:
[n,mn] = size(A);
% A_O = A;
m = numel(b);
fprintf('shape A:, %d,%d, b size: %d', n, mn, m);
A_ori = A;
A = reshape(A, n^2, m)';
nb = norm(b);
nC = normest(C);

%% initialization, all vectors;
X = sqrt(n)* speye(n);
Z = sqrt(n) * speye(n);
y = sqrt(n) * ones(m, 1);

fprintf('\n size of C: %s , m: %d, n:%d \n', num2str(size(C)), m, n);

sigma = 0.2;

MAX_ITER = 100;
AA = A * A';
p = symamd(AA);

zerosn = sparse(n^2, 1);
zerosm = sparse(m, 1);
phi_m = 2.2e-16 * 1 * speye(m);
phi_n =  2.2e-16 * 1 * speye(n);

fprintf('Iter. No     P_inf     D_inf     Gap      time\n');
fprintf('----------------------------------------------\n');
% M = cell(m, m);
% M = zeros(m,m);
MV = zeros(n^2, m);
for iter=1:MAX_ITER
    
    st = cputime;
    
    rd = C(:) - Z(:) - A'*y;
    Rd = reshape(rd, n, n);
    rp = b-A*X(:);
    xz = X(:).*Z(:);
    mu = sum(xz)/n;
    
    res_primal = norm(rp)/(1 + nb);
    res_dual = norm(rd)/(1 + nC);
    gap = full(mu);
    %% predictor:
    Z_inv = inv(Z);
    ZRX =  Z_inv * Rd * X;
    Rc = 2 * (sigma * mu * Z - Z * X * Z);
    
    ZA_ori = Z_inv * A_ori;
    for i=1:m
        XZAi = ZA_ori(:,(i-1)*n + 1:(i-1)*n + n) * X;
        MV(:, i) = XZAi(:);
    end
%     M = A * reshape(XAZ', n^2, m);
    M = A * MV;
%     M = symm(M + phi_m);
    [dX, dZ, dy] = direction(rd, rp, Rc);
%     
    %% corrector step:
%     
%     alpha_X = max_psd_stepsize(X, dX, -1.5);
%     alpha_Z = max_psd_stepsize(Z, dZ, -1.5);
%     
%     X_p = X+alpha_X*dX;
%     Z_p = Z+alpha_Z*dZ;
%     
%     sigma = (X_p(:)'*Z_p(:)/sum(xz))^3;
%     sigma = min(0.2, sigma);
% 
%     ZXZ = Z_p * X_p * Z_p;
%     Rc = 0.2 * mu * Z_p - ZXZ - Z_p * X_p * Z_p;
%     [dX2, dZ2, dy2] = direction(zerosn, zerosm, Rc);
%     
%     %% combine:
%     dX = dX + dX2;
%     dZ = dZ + dZ2;
%     dy = dy + dy2;

    %% refinement:
    dx = dX(:) - A'*((AA)\(A*(X(:)+dX(:))-b));
    dX = reshape(dx, n, n); 

    %% update variables:
    alpha_X = max_psd_stepsize(X, dX, -1.2);
    alpha_Z = max_psd_stepsize(Z, dZ, -1.2);

    X = X + alpha_X * dX + phi_n;
    Z = Z + alpha_Z * dZ + phi_n;
    y = y + alpha_Z * dy;
    
    %% stopping criteria for primal, dual, duality:
    elps = cputime - st;
    fprintf('iter %3d:  %5.2e, %5.2e, %5.2e, %2.2f\n', iter, res_primal, res_dual, gap, elps);
    if res_primal < tol && res_dual < tol && gap < tol
        disp("Converged!");
        out.cputime = elps;
        out.optval = C(:)' * X(:);
        out.solver = 'mysdp';
        out.slvitr = iter;
        out.slvtol = tol;
        break
    end
    
end

    %% get direction
    function [dX, dZ, dy] = direction(rd, rp, Rc)
        %% calculate dy, using chol
        rhss = ZRX + ZRX' - Z_inv * Rc * Z_inv;
        rhs = rp + A * (rhss(:) / 2);
        R = chol(M(p, p));
        dy_p(p) = R\(R'\rhs(p));
        dy = dy_p';
        %% calculate dZ
        dz = rd - A' * dy;
        dZ = reshape(dz, n, n);
        dZ = symm(dZ);
        %% calculate dX
        xdz = X * dZ * Z_inv;
        dX = sigma * mu * Z_inv - X - (xdz+ xdz')/2;
        dX = symm(dX);
        
    end

end


%% get max stepsize obtaining psd
function [alpha] = max_psd_stepsize(V, dV, mini)
    % using chol to check psd.
    alpha_max  = -1/min(min(diag(dV)./diag(V)), mini);
    alpha = alpha_max;
    [R, is_sdp] = chol(V+alpha*dV);
    while is_sdp > 0
        alpha = alpha - alpha_max * 0.1;
        [R, is_sdp] = chol(V+alpha*dV);
    end
end

function [H] = symm(H)
    H = (H + H')/2;
end
