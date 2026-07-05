clc; clear variables; close all;

% SNR range in dB
Pt = -114:5:-54;    % Transmit power in dB
pt = db2pow(Pt);    % Convert dB to linear scale

N = 10^4; % Number of samples

% Distances of users
d1 = 20; d2 = 10; d3 = 5; d4 = 3;
eta = 3.5; % Path loss exponent

% Rayleigh fading coefficients for users
h1 = (sqrt(d1^-eta)) * (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
h2 = (sqrt(d2^-eta)) * (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
h3 = (sqrt(d3^-eta)) * (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);
h4 = (sqrt(d4^-eta)) * (randn(N, 1) + 1i * randn(N, 1)) / sqrt(2);

% Channel gains
g1 = abs(h1).^2;
g2 = abs(h2).^2;
g3 = abs(h3).^2;
g4 = abs(h4).^2;

BW = 10^6; % Bandwidth
No = -174 + 10 * log10(BW); % Noise power in dBm
no = (10^-3) * db2pow(No); % Noise power in linear scale

% Power allocation coefficients for Hybrid NOMA
a1 = 0.7; a2 = 1 - a1;
b1 = 0.7; b2 = (3/4) * (1 - b1); b3 = (3/4) * (1 - (b1 + b2)); b4 = 1 - (b1 + b2 + b3);

% Target data rates (bps/Hz)
R_near = 0.5; R_far = 0.5;
rate_near = 2^R_near - 1;
rate_far = 2^R_far - 1;

% Pre-allocate outage probability arrays
pout_near = zeros(size(Pt));
pout_far = zeros(size(Pt));
pout_sc_noma = zeros(size(Pt));

for u = 1:length(pt)
    % Calculate SNRs for Near-Near pairing
    gamma1_near = pt(u) * a1 .* g1 ./ (pt(u) * (a2) .* g1 + no);
    gamma2_near = pt(u) * a1 .* g2 ./ (pt(u) * (a2) .* g2 + no);
    gamma3_near = pt(u) * a1 .* g3 ./ (pt(u) * (a2) .* g3 + no);
    gamma4_near = pt(u) * a1 .* g4 ./ (pt(u) * (a2) .* g4 + no);
    
    % Calculate SNRs for Near-Far pairing
    gamma1_far = pt(u) * b1 .* g1 ./ (pt(u) * (b2 + b3 + b4) .* g1 + no);
    gamma2_far = pt(u) * b2 .* g2 ./ (pt(u) * (b3 + b4) .* g2 + no);
    gamma3_far = pt(u) * b3 .* g3 ./ (pt(u) * b4 .* g3 + no);
    gamma4_far = pt(u) * b4 .* g4 ./ no;
    
    % Calculate SNRs for SC-NOMA (Successive Cancellation NOMA)
    gamma1_sc = pt(u) * b1 .* g1 ./ (pt(u) * (b2 + b3 + b4) .* g1 + no);
    gamma2_sc = pt(u) * b2 .* g2 ./ (pt(u) * (b3 + b4) .* g2 + no);
    gamma3_sc = pt(u) * b3 .* g3 ./ (pt(u) * b4 .* g3 + no);
    gamma4_sc = pt(u) * b4 .* g4 ./ no;

    % Calculate achieved rates for Near-Near pairing
    R1_near = log2(1 + gamma1_near);
    R2_near = log2(1 + gamma2_near);
    R3_near = log2(1 + gamma3_near);
    R4_near = log2(1 + gamma4_near);
    
    % Calculate achieved rates for Near-Far pairing
    R1_far = log2(1 + gamma1_far);
    R2_far = log2(1 + gamma2_far);
    R3_far = log2(1 + gamma3_far);
    R4_far = log2(1 + gamma4_far);

    % Calculate achieved rates for SC-NOMA
    R1_sc = log2(1 + gamma1_sc);
    R2_sc = log2(1 + gamma2_sc);
    R3_sc = log2(1 + gamma3_sc);
    R4_sc = log2(1 + gamma4_sc);
    
    % Check for outage for Near-Near pairing
    for k = 1:N
        if (R1_near(k) < R_near) || (R2_near(k) < R_near) || (R3_near(k) < R_near) || (R4_near(k) < R_near)
            pout_near(u) = pout_near(u) + 1;
        end
    end
    
    % Check for outage for Near-Far pairing
    for k = 1:N
        if (R1_far(k) < R_far) || (R2_far(k) < R_far) || (R3_far(k) < R_far) || (R4_far(k) < R_far)
            pout_far(u) = pout_far(u) + 1;
        end
    end
    
    % Check for outage for SC-NOMA
    for k = 1:N
        if (R1_sc(k) < R_far) || (R2_sc(k) < R_far) || (R3_sc(k) < R_far) || (R4_sc(k) < R_far)
            pout_sc_noma(u) = pout_sc_noma(u) + 1;
        end
    end
end

% Normalize outage probabilities
pout_near = pout_near / N;
pout_far = pout_far / N;
pout_sc_noma = pout_sc_noma / N;




SNR = Pt - No; % In dB
% Plotting
figure;
semilogy(SNR, pout_near, 'o-m', 'linewidth', 1.5); hold on; grid on;
semilogy(SNR, pout_far, 'o-k', 'linewidth', 1.5);
semilogy(SNR, pout_sc_noma, 'o-b', 'linewidth', 1.5);
xlabel('Transmit power (dBm)');
ylabel('Outage probability');
title('Outage Probability vs Transmit Power for Hybrid NOMA');
legend('Near-Near Pairing', 'Near-Far Pairing', 'SC-NOMA');
