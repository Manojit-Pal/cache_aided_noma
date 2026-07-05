clc; clear variables; close all;

%SNR range
Pt = -114:5:-54;	%in dB
pt = db2pow(Pt);	%in linear scale

N = 10^4;

d1 = 20; d2 = 10; d3 =5; d4=3;      %Distance of users
eta = 3.5;                              %Path loss exponent

%Rayleigh fading coefficients of both users
h1 = (sqrt(d1^-eta))*(randn(N,1) + 1i*randn(N,1))/sqrt(2);
h2 = (sqrt(d2^-eta))*(randn(N,1) + 1i*randn(N,1))/sqrt(2);
h3 = (sqrt(d3^-eta))*(randn(N,1) + 1i*randn(N,1))/sqrt(2);
h4 = (sqrt(d4^-eta))*(randn(N,1) + 1i*randn(N,1))/sqrt(2);
%Channel gains
g1 = (abs(h1)).^2;
g2 = (abs(h2)).^2;
g3 = (abs(h3)).^2;
g4 = (abs(h4)).^2;

BW = 10^6;
No = -174 + 10*log10(BW);
no = (10^-3)*db2pow(No);

%Power allocation coefficients
%a1 = 0.85; a2 = 0.15; a3=0.1; a4 = 1-(a1+a2+a3);
a1 = 0.7; a2 = (3/4)*(1-a1); a3 = (3/4)*(1-(a1+a2)); a4 = 1 - (a1+a2+a3);

%C_noma = zeros(1,length(pt));
%C_oma = zeros(1,length(pt));
for u = 1:length(pt)
    
    %NOMA capacity calculation 4 USER
    C_noma_1 = log2(1 + pt(u)*a1.*g1./(pt(u)*a2.*g1 + pt(u)*a3.*g1+pt(u)*a4.*g1 + no));      %User 1
    C_noma_2 = log2(1 + pt(u)*a2.*g2./(pt(u)*a3.*g2+pt(u)*a4.*g1+no));                       %User 2
    C_noma_3 = log2(1 + pt(u)*a3.*g3./(pt(u)*a4.*g3+no));                                     %USER 3
    C_noma_4 = log2(1 + pt(u)*a4.*g4/no);                                                     %User 4
    C_noma_sum(u) = mean(C_noma_1 + C_noma_2 + C_noma_3+C_noma_4);  %Sum capacity of NOMA
    
    %OMA capacity calculation
    C_oma_1 = (1/3)*log2(1 + pt(u)*g1/no);    %User 1
    C_oma_2 = (1/3)*log2(1 + pt(u)*g2/no);    %User 2
    C_oma_3 = (1/3)*log2(1 + pt(u)*g3/no);    %User 3
    C_oma_4= (1/3)*log2(1 + pt(u)*g4/no);    %User 4
    
    C_oma_sum(u) = mean(C_oma_1 + C_oma_2 + C_oma_3+ C_oma_4); %Sum capacity of OMA
    
    
     %1-2  3-4 (n-n, f-f)
   C1 = (log2(1 + a1.*g1*pt(u)./(a2.*g1*pt(u)+no)));
   C2 = (log2(1 + a2.*g2*pt(u)./no));
   C3 = (log2(1 + a1.*g3*pt(u)./(a2.*g3*pt(u)+no)));
   C4 = (log2(1 + a2.*g4*pt(u)./no));
   
   R1_NNFF_sum(u) = 0.5*(mean(C1+C2+C3+C4));
    
%    %1-4  2-3 (n-f)
    C11 = (log2(1 + a1.*g1*pt(u)./(a2.*g1*pt(u)+no)));
    C44 = (log2(1 + a2.*g4*pt(u)./no));
    C22 = (log2(1 + a1.*g2*pt(u)./(a2.*g2*pt(u)+no)));
    C33 = (log2(1 + a2.*g3*pt(u)./no));
  
   R2_NF_sum(u) = 0.5*(mean(C11+C22+C33+C44));
       
%    %TDMA
%    t1(u) = 0.25*mean(log2(1 + g1*snr(u)));
%    t2(u) = 0.25*mean(log2(1 + g2*snr(u)));
%    t3(u) = 0.25*mean(log2(1 + g3*snr(u)));
%    t4(u) = 0.25*mean(log2(1 + g4*snr(u)));
%    
% %    %NOMA
%    n1(u) = mean(log2(1 + b1.*g1*snr(u)./((b2+b3+b4).*g1*snr(u)+1)));
%    n2(u) = mean(log2(1 + b2.*g2*snr(u)./((b3+b4).*g2*snr(u)+1)));
%    n3(u) = mean(log2(1 + b3.*g3*snr(u)./(b4.*g3*snr(u)+1)));
%    n4(u) = mean(log2(1 + b4.*g4*snr(u)));
%    
% %   
end
    
    
SNR = Pt - No; %In DB the SNR = Power - Noise in liner scale Pt/No
figure;


plot(SNR,C_noma_sum,'o-b','linewidth',1.5);hold on; grid on;
%plot(SNR,C_oma_sum,'o-r','linewidth',1.5); %oma 
plot(SNR,R1_NNFF_sum,'o-m','linewidth',1.5);
plot(SNR,R2_NF_sum,'o-k','linewidth',1.5);

legend('SC- NOMA','Hybrid NOMA N-N,F-F pairing','Hybrid NOMA N-F pairing');
%legend('Hybrid NOMA N-F pairing','Hybrid NOMA N-N,F-F pairing','SC-NOMA','TDMA');
xlabel('SNR (dB)');
ylabel('Sum rate (bps/Hz)');





p = length(SNR);
p1 = zeros(1,length(SNR));
p2 = zeros(1,length(SNR));
p3 = zeros(1,length(SNR));

R1= 0.5; % target data rate bps/HZ
R2=.5;    % taeget data rate bps/HZ

rate1 = 2^R1 -1; % target SNR for far user
rate2= 2^R2 -1;  % target SNR for near user
rate3= 2^R2 -1;  % target SNR for near user
       %Target rate of users in bps/Hz
for u = 1:p
%        Calculate SNRs
    gamma_1 = log2(1+pt(u)*a1.*g1./(pt(u)*(a2+a3)*g1 + no)) ;            % SNR for Far user at D1=500 distance point
    gamma_12 = log2(1 + pt(u)*a1.*g2./(pt(u)*(a2+a3)*g2 + no));  % SNR for Near User at D1=500 Distance Point
    gamma_13= log2(1 + pt(u)*a1.*g3./(pt(u)*(a3)*g3 + no));   % SNR for Near User at D1=500 Distance Point
    gamma_2 =  pt(u)*a2.*g2./(pt(u)*(a3)*g2 + no);               % SNR for Near User at D2=120 m Distance Point
    gamma_23= pt(u)*a2.*g3./(pt(u)*(a3)*g3 + no);
    gamma_3= pt(u)*a3.*g3./no;
    
%      gamma_1 = a1*pt(u)*g1./(a2*pt(u)*g1+no);  % SNR for Far user at D1=1000 distance point
%     gamma_12 = a1*pt(u)*g2./(a2*pt(u)*g2+no); % SNR for Near User at D1=1000 Distance Point
%     gamma_2 = a2*pt(u)*g2/no;                 % SNR for Near User at D2=500 m Distance Point
    
    
%     %Calculate achievable rates
     R1 = log2(1+gamma_1);
     R12 = log2(1+gamma_12);
     R13 = log2(1+gamma_13);
     R2 = log2(1+gamma_2);
     R23 = log2(1+gamma_23); 
     R3 = log2(1+gamma_3);
    
      %NOMA capacity calculation 4 USER
%     C_noma_1 = log2(1 + pt(u)*a1.*g1./(pt(u)*(a2+a3+a4)*g1 + no));      %Capacity Farest 500 mUser 1
%      C_noma_12 = log2(1 + pt(u)*a1.*g2./(pt(u)*(a2+a3+a4)*g2 + no)); % Capacity for user 2 at 120 m point
%     C_noma_2 = log2(1 + pt(u)*a2.*g2./(pt(u)*(a3+a4)*g2 + no));          %User 2 120m 
%      C_noma_12 = log2(1 + pt(u)*a1.*g2./(pt(u)*a3.*g2+pt(u)*a4.*g1+no)); 
     
     
%     C_noma_3 = log2(1 + pt(u)*a3.*g2./(pt(u)*a4.*g1+no));                                     %USER 3
%     C_noma_34 = log2(1 + pt(u)*a3.*g4./(pt(u)*a4.*g4+no));                                     %USER 34
%     C_noma_4 = log2(1 + pt(u)*a4.*g4/no);                                                     %Uesr 4
    
    %Find average of achievable rates
     R1_av(u) = mean(R1 );
     R2_av(u) = mean(R2);
     R3_av(u) = mean(R3);
     R12_av(u) = mean(R12);
     R23_av(u) = mean(R23);
     R13_av(u) = mean(R13);
    
    %Check for outage
    for k = 1:N
        if R1(k) < rate1
            p1(u) = p1(u)+1;
        end
        if (R12(k) < rate1)||(R2(k) < rate2)
            p2(u) = p2(u)+1;
        end
        if (R12(k) < rate1)|| (R23(k)<rate2) ||(R3(k) < rate3)
            p3(u) = p3(u)+1;
        end
    
    end
end

pout1 = p1/N; 
pout2 = p2/N;
pout3 = p3/N;

figure;
semilogy(SNR, pout1, 'linewidth', 1.5); hold on; grid on;
semilogy(SNR, pout2, 'linewidth', 1.5);
semilogy(SNR, pout3, 'linewidth', 1.5);
xlabel('Transmit power (dBm)');
ylabel('Outage probability');
legend('User 1 (far user)','User 2 (near user)','User 2 (near user)');

% % figure;
% % plot(SNR, R1_av, 'linewidth', 1.5); hold on; grid on;
% % plot(SNR, R12_av, 'linewidth', 1.5);
% % plot(SNR, R2_av, 'linewidth', 1.5);
% xlabel('Transmit power (dBm)');
% ylabel('Achievable capacity (bps/Hz)');
% legend('R_1','R_{12}','R_2')
% 
% 
