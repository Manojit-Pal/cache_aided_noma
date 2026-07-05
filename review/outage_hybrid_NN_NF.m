
%Biswajit Ghosh hybrid NOMA%
%Outage probability for NOMA using m-Nakagami fading coefficient%
clc; clear variables; 
close all;
d1 = 20; d2 = 15;
d3 = 10; d4 = 5;
eta = 4;
N = 10^3;
BW = 10^6;
No = -174 + 10*log10(BW);
no = (10^-3)*db2pow(No);
w = 3;                              % Same as omega above %Parameters for m-Nakagami Fading 
x = 0:0.05:3;                       %Parameters for m-Nakagami Fading
m =1;                               %( Nakagami fading parameter m) %Parameters for m-Nakagami Fading
SNR = 20:2:65;
snr = db2pow(SNR);


% Target data rates in bps/Hz
R1_target = 0.5; 
R2_target = 0.5; 
R3_target = 0.5; 
R4_target = 0.5;

rate1 = 2^R1_target - 1;
rate2 = 2^R2_target - 1;
rate3 = 2^R3_target - 1;
rate4 = 2^R4_target - 1;

% Initialize outage probability arrays
p1 = zeros(1, length(snr));
p2 = zeros(1, length(snr));
p3 = zeros(1, length(snr));
p4 = zeros(1, length(snr));

R1_NNFF_outage = zeros(1, length(snr));
R2_NF_outage = zeros(1, length(snr));

for ii = 1:length(x)
    
NakL(ii)=((2*m^m)/(gamma(m)*w^m))*x(ii)^(2*m-1)*exp(-((m/w)*x(ii)^2)); %PDF of Nakagami fading

mu=1;          
eta1=2;           
% h=(2+(1/eta)+eta)/4;                                                        %For format 1
% H=((1/eta)-eta)/4;                                                          %For format 1
% mu_p=(mu+1/2);
% mul=(mu-1/2);
% upper_term= (4*sqrt(pi)*mu^mu_p)*h^mu;                                                                             %outage prob 
% lower_term= gamma(mu)*H^mul;
% t11=upper_term/lower_term;
% t12=x(ii).^(2*mu);
% besy= 2*mu*H.*x(ii).^2;
% t13=besseli((mu-(1/2)),besy);
% t14=exp(-2*mu*h.*x(ii).^2);
h=(2+(1/eta1)+eta1)/4;                                                        %For format 1
H=((1/eta1)-eta1)/4;                                                          %For format 1
mu_p=(mu+1/2);
mul=(mu-1/2);
upper_term= (4*sqrt(pi)*mu^mu_p)*h^mu;                                                                             %outage prob 
lower_term= gamma(mu).*H^mul;
t11=upper_term./lower_term;
t12=x.^(2*mu);
besy= 2*mu*H.*x.^2;
t13=besseli((mu-(1/2)),besy);
t14=exp(-2*mu*h.*x.^2);

pdf_eta=(t12.*t14.*t13).*t11; 

 h1 = sqrt(d1^-eta).*pdf_eta/sqrt(2); % fading based on Eta mu fading channel
 h2 = sqrt(d2^-eta).*pdf_eta/sqrt(2);  % fading based on Eta mu fading channel
 h3 = sqrt(d3^-eta).*pdf_eta/sqrt(2);   % fading based on Eta mu fading channel
 h4 = sqrt(d4^-eta).*pdf_eta/sqrt(2);   % fading based on Eta mu fading channel
 
 
 h11 = sqrt(d1^-eta).*NakL/sqrt(2);   % fading based on Nakagami fading channel
 h21 = sqrt(d2^-eta).*NakL/sqrt(2);    % fading based on Nakagami fading channel
 h31 = sqrt(d3^-eta).*NakL/sqrt(2);     % fading based on Nakagami fading channel
 h41 = sqrt(d4^-eta).*NakL/sqrt(2);    % fading based on Nakagami fading channel

 

end
g1 = (abs(h1)).^2;            % fading based on Eta mu fading channel
g2 = (abs(h2)).^2;            % fading based on Eta mu fading channel
g3 = (abs(h3)).^2;            % fading based on Eta mu fading channel
g4 = (abs(h4)).^2;            % fading based on Eta mu fading channel

g11 = (abs(h11)).^2;            % fading based on Nakagami fading channel
g21 = (abs(h21)).^2;            % fading based on Nakagami fading channel
g31 = (abs(h31)).^2;            % fading based on Nakagami fading channel
g41 = (abs(h41)).^2;            % fading based on Nakagami fading channel





a1 = 0.7; a2 = 1-a1;
b1 = 0.7; b2 = (3/4)*(1-b1); b3 = (3/4)*(1-(b1+b2)); b4 = 1 - (b1+b2+b3);
T = 1;
epsilon = (2^T)-1;

for u = 1:length(snr)
   
   %1-2  3-4 (n-n, f-f)
  
   C1(u) = mean(log2(1 + a1.*g1*snr(u)./(a2.*g1*snr(u)+1)));
   C2(u) = mean(log2(1 + a2.*g2*snr(u)));
   C3(u) = mean(log2(1 + a1.*g3*snr(u)./(a2.*g3*snr(u)+1)));
   C4(u) = mean(log2(1 + a2.*g4*snr(u)));
   
   C1_naka(u) = mean(log2(1 + a1.*g11*snr(u)./(a2.*g11*snr(u)+1)));
   C2_naka(u) = mean(log2(1 + a2.*g21*snr(u)));
   C3_naka(u) = mean(log2(1 + a1.*g31*snr(u)./(a2.*g31*snr(u)+1)));
   C4_naka(u) = mean(log2(1 + a2.*g41*snr(u)));
   
   %1-4  2-3 (n-f)
   C11(u) = mean(log2(1 + a1.*g1*snr(u)./(a2.*g1*snr(u)+1)));
   C44(u) = mean(log2(1 + a2.*g4*snr(u)));
   C22(u) = mean(log2(1 + a1.*g2*snr(u)./(a2.*g2*snr(u)+1)));
   C33(u) = mean(log2(1 + a2.*g3*snr(u)));
   
   C11_naka(u) = mean(log2(1 + a1.*g11*snr(u)./(a2.*g11*snr(u)+1)));
   C44_naka(u) = mean(log2(1 + a2.*g41*snr(u)));
   C22_naka(u) = mean(log2(1 + a1.*g21*snr(u)./(a2.*g21*snr(u)+1)));
   C33_naka(u) = mean(log2(1 + a2.*g31*snr(u)));
   
   
%    %TDMA
%    t1(u) = 0.25*mean(log2(1 + g1*snr(u)));
%    t2(u) = 0.25*mean(log2(1 + g2*snr(u)));
%    t3(u) = 0.25*mean(log2(1 + g3*snr(u)));
%    t4(u) = 0.25*mean(log2(1 + g4*snr(u)));
%    
%    t1_naka(u) = 0.25*mean(log2(1 + g11*snr(u)));
%    t2_naka(u) = 0.25*mean(log2(1 + g21*snr(u)));
%    t3_naka(u) = 0.25*mean(log2(1 + g31*snr(u)));
%    t4_naka(u) = 0.25*mean(log2(1 + g41*snr(u)));
%   
   
   %NOMA
   n1(u) = mean(log2(1 + b1.*g1*snr(u)./((b2+b3+b4).*g1*snr(u)+1)));
   n2(u) = mean(log2(1 + b2.*g2*snr(u)./((b3+b4).*g2*snr(u)+1)));
   n3(u) = mean(log2(1 + b3.*g3*snr(u)./(b4.*g3*snr(u)+1)));
   n4(u) = mean(log2(1 + b4.*g4*snr(u)));
   
   n1_naka(u) = mean(log2(1 + b1.*g11*snr(u)./((b2+b3+b4).*g11*snr(u)+1)));
   n2_naka(u) = mean(log2(1 + b2.*g21*snr(u)./((b3+b4).*g21*snr(u)+1)));
   n3_naka(u) = mean(log2(1 + b3.*g31*snr(u)./(b4.*g31*snr(u)+1)));
   n4_naka(u) = mean(log2(1 + b4.*g41*snr(u)));
   
   
   
   
   % Calculate outage probability for NOMA and TDMA hybrid NOMA
   % pout_noma(u) = sum(C_noma_1 < log2(1 + (2^1 - 1))) / N; % Considering User 1's outage
    %pout_hybrid(u) = sum(C_tdma_hybrid < log2(1 + (2^1 - 1))) / N; % Considering any user's outage
    
    % Check for outage in hybrid NOMA schemes
    
    
    
    
    R1_NNFF_outage(u) = mean(C1_naka < rate1 | C2_naka < rate2 | C3_naka < rate3 | C4_naka < rate4);
    R2_NF_outage(u) = mean(C11_naka < rate1 | C22_naka < rate2 | C33_naka < rate3 | C44_naka < rate4);
   
 %   
end
R1 = 0.5*(C1+C2+C3+C4);
R1_naka = 0.5*(C1_naka+C2_naka+C3_naka+C4_naka);

R11 = 0.5*(C11+C22+C33+C44);

R11_naka = 0.5*(C11_naka+C22_naka+C33_naka+C44_naka);

% t = (t1+t2+t3+t4);
% 
 t_naka = (t1_naka+t2_naka+t3_naka+t4_naka);

n = n1+n2+n3+n4;

n_naka = n1_naka+n2_naka+n3_naka+n4_naka;
figure;
 
 plot(SNR,R11_naka,'*-b','linewidth',1.5);hold on; grid on;
 plot(SNR,R1_naka,'*-r','linewidth',1.5); 
 plot(SNR,n_naka,'*-m','linewidth',1.5);
 plot(SNR,t_naka,'*-k','linewidth',1.5);


%semilogy(SNR, R1_NNFF_outage, 'linewidth', 2);
%semilogy(SNR, R2_NF_outage, 'linewidth', 2);
xlabel('SNR (dB)');
ylabel('End to End Outage probability');


% plot(SNR,R11,'o-b','linewidth',1.5);hold on; grid on;
% plot(SNR,R1,'o-r','linewidth',1.5); 
% plot(SNR,n,'o-m','linewidth',1.5);
%  plot(SNR,t,'o-k','linewidth',1.5);

legend('Hybrid NOMA NF Nakagami-m(1)','Hybrid NOMA NN-FF pairing-Nakagami-m(1)','SC-NOMA-Nakagami-m(1)Fading','Hybrid NOMA NF pairing-\eta=2-\mu=1','Hybrid NOMA NN-FF pairing-\eta=2-\mu=1','SC-NOMA-\eta=2-\mu=1');
xlabel('SNR (dB)');
ylabel('Sum rate (bps/Hz)');












