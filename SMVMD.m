%**************************************************************************
%  Copyright(c) 2021 Shuaishuai Liu. All rights reserved.                 *
%**************************************************************************
%  Shuaishuai Liu, Ph.D.                                                  *
%  Department of Astronautic Science and Mechanics                        *
%  Harbin Institute of Technology, China                                  *
%  E-mail: liushuaishuai_hit@163.com                                      *
%**************************************************************************
%  Runtime environment: MATLAB R2016b or later                            *
%**************************************************************************
%  Introduction:                                                          *
%                                                                         *
%**************************************************************************
% If you use this program to do productive scientific research that leads *
% to publication, we ask that you acknowledge use of the program by       *
% citing the following papers in your publication:                        *
% [1] Shuaishuai Liu, Kaiping Yu*. Successive multivariate variational    *
%     mode decomposition based on instantaneous linear mixing model.      *
%     Signal Processing, 2022, 190:108311.                                *
% [2] Shuaishuai Liu, Kaiping Yu*. Successive multivariate variational    *
%     mode decomposition. Multidimensional Systems and Signal Processing. *
%     2022. DOI: 10.1007/s11045-022-00828-w.                              *
%**************************************************************************

function [u, u_hat, omega_iter]=SMVMD(signal, alphaMin, alphaMax, beta, init, tau, eps1, eps2, K)

% Successive Multivariate Variational Mode Decomposition
% Input and Parameters:
% ---------------------
% signal    - input multivariate signal that needs to be decomposed
% alphaMin  - the min parameter that defines the bandwidth of extracted modes 
% alphaMax  - the max parameter that defines the bandwidth of extracted modes 
%             (low value of alpha yields higher bandwidth)
% beta      - the change rate of alpha > 1
% init      - 0 = the first omega start at 0
%           - 1 = the first omega initialized randomly
% tau       - time-step of the dual ascent ( pick 0 for noise-slack )
% eps1      - tolerance value for convergence of ADMM
% eps2      - tolerance value for convergence of ternimation
% K        - the maximum of mode number 
%
%
% Output:
% ----------------------
% u       - the collection of decomposed modes
% u_hat   - spectra of the modes
% omega   - estimated mode center-frequencies

%% Check for getting number of channels from input signal===========
[row, col] = size(signal);
if row > col
	C = col;% number of channels
    T = row;% length of the Signal
	signal = signal';
else
	C = row;% number of channels
    T = col;% length of the Signal
end
%% ---------- Preparations==========================================
% Mirroring
flag = false;
if mod(T,2)==0
    flag = true;
    f(:,1:T/2) = -signal(:,T/2:-1:1);
    f(:,T/2+1:3*T/2) = signal;
    f(:,3*T/2+1:2*T) = -signal(:,T:-1:T/2+1);
else
    f(:,1:(T-1)/2) = -signal(:,(T-1)/2:-1:1);
    f(:,(T+1)/2+1:(3*T+1)/2) = signal;
    f(:,(3*T+1)/2+1:2*T) = -signal(:,T:-1:(T+1)/2+1);    
end

% Time Domain 0 to T (of mirrored signal)
T = size(f,2);
t = (1:T)/T;

% frequency points
fp = t-0.5-1/T;
fp = fp(T/2+1:end);

% Construct and center f_hat
f_hat = fftshift(fft(f,[],2),2);
f_hat_plus = f_hat(:,T/2+1:end);

%% ------------ �ⲿ��ʼ��===========================================
u_hat_plus = zeros(K,T/2,C);    % �洢ģʽ��Ƶ����
omega = zeros(1,K);             % ����Ƶ�ʴ洢 
omega_iter = cell(1,K);         % ����Ƶ�ʵ������̴洢 
k = 0;                          % ģʽ����
sum_uk = 0;                     % ǰk-1��ģʽ��
sum_fuk = 0;                    % ǰk-1���˲����Ӻ�
tol2 = eps2+eps;                % �㷨��ֹ��ֵ

%% ----------- ִ���㷨���==========================================
while (tol2 > eps2 && k<K)
    
    k = k+1;    % ģʽ����һ
    
    %% -----------ģʽk��ȡ��ʼ����������=============================
	% ��ʼ������Ƶ��omega
    switch init
        case 0
            omega(k) = 0.25;        % ��һ��Ƶ���е��ʼ��
        otherwise
            omega(k) = rand(1)/2.56; % �����ʼ��   
    end
    omega_iter{k}(1) = omega(k); % ��¼k��ģʽ����Ƶ�ʳ�ʼֵ

    alpha = alphaMin;   % ��ʼ������Ʋ���alpha
    lambda_hat = zeros(C,T/2);     % ��ʼ���������ճ���
    tol1 = eps1+eps;    % ģʽk������ֹ��ֵ
    n = 1;              % ��ʼ������������
    uk_hat_c = zeros(C,T/2); % ��ʼ����ǰģʽƵ����
    uk_hat_n = zeros(C,T/2); % ��ʼ����һ��ģʽƵ����
    filt_uk = alpha*(fp - omega(k)).^2; % ��ʼ����k��ģʽ�˲���
    
    %% ----------- ����������ȡ��k��ģʽ
    while (tol1>eps1 && n<100)
        % ���µ�k�������ģʽ
        for c=1:C
            uk_hat_n(c,:) = (f_hat_plus(c,:)+(filt_uk.^2).*uk_hat_c(c,:)+lambda_hat(c,:)/2)./...
                ((1+(filt_uk.^2)).*(1+2*filt_uk+sum_fuk));
        end

        % ��������Ƶ��
        numerator = sum(fp*abs(uk_hat_n').^2);
        denominator = sum(abs(uk_hat_n(:)).^2);
        omega(k) = numerator/denominator;

        filt_uk = alpha*(fp - omega(k)).^2;
            
        % Dual ascent
        lambda_hat = lambda_hat + tau*((f_hat_plus-uk_hat_n+lambda_hat/2)./...
                (1+repmat(filt_uk.^2,C,1))-lambda_hat/2);

        % converged yet?
        tol1 =  sum(abs(uk_hat_n(:)-uk_hat_c(:)).^2)/(sum(abs(uk_hat_c(:)).^2)+eps);
        uk_hat_c = uk_hat_n;
            
        alpha = min(alpha*beta,alphaMax);
        
        n = n+1;        % ѭ����������һ
        omega_iter{k}(n) = omega(k); % ��¼k��ģʽ����Ƶ�ʵ�������
    end
    
    sum_uk = sum_uk+uk_hat_c;
    sum_fuk = sum_fuk+1./(filt_uk.^2);
    u_hat_plus(k,:,:) = uk_hat_c.';
    
    % converged yet?
    tol2=sum(abs((f_hat_plus(:)-sum_uk(:)).^2))/sum(abs(f_hat_plus(:)).^2);
%     display(['Extracting Mode: ' int2str(k) ' Update times: ' int2str(n)])

end

%% ------ Post-processing and cleanup
% discard the last item, which maybe a noise item
K = k;
omega = omega(1:K);
omega_iter=omega_iter(1:K);
u_hat_plus = u_hat_plus(1:K,:,:);

% Signal reconstruction
u_hat = zeros(K, T, C);
u_hat(:,(T/2+1):T,:) = u_hat_plus;
u_hat(:,(T/2+1):-1:2,:) = conj(u_hat_plus);
u_hat(:,1,:) = conj(u_hat(:,end,:));

    
u = zeros(K,T,C);
for k = 1:K
	for c = 1:C
		u(k,:,c)=real(ifft(ifftshift(u_hat(k,:,c))));
	end
end
% remove mirror part
if flag
    u = u(:,T/4+1:3*T/4,:);
else
    u = u(:,(T+2)/4+1:(3*T+2)/4,:);    
end

% recompute spectrum
clear u_hat;
for k = 1:K
	for c = 1:C
		u_hat(k,:,c)=fftshift(fft(u(k,:,c)))';
	end
end
