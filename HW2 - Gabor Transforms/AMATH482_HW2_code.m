clear all; close all; clc;

%%% Name: ROSEMICHELLE MARZAN
%%% Course: AMATH 482
%%% Homework 2, Due 2/7/2019

%% PART I: Spectrograms for Handel's 'Messiah'

load handel

% Defining the domain
S = y';                            % signal
L = length(S)/Fs;                  % length of sample
n = length(S);                     % # of data points
t = linspace(0,L,n);               % vector of time points
k = (1/L)*[0:(n-1)/2 -(n-1)/2:-1]; % vector of frequencies
ks = fftshift(k);

% VARYING WINDOW SIZE (Gaussian)
a = 1000; dtau = 0.1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    g = exp(-a*(t - tslide(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
colormap(hot) 
ylim([0 2000])
title(['Messiah spectrogram, a = ', num2str(a), ', d\tau = ', num2str(dtau)])
xlabel('time (s)')
ylabel('frequency (Hz)')


% OVERSAMPLING AND UNDERSAMPLING (Gaussian)
a = 1000; dtau = 1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    g = exp(-a*(t - tslide(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
colormap(hot)
ylim([0 2000])
title(['Messiah spectrogram, a = ', num2str(a), ', d\tau = ', num2str(dtau)])
xlabel('time (s)')
ylabel('frequency (Hz)')


% MEXICAN HAT WINDOW
a = 10; dtau = 1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    % Mexican Hat filter function
    g = (1 - (a*(t - tslide(j))).^2).*exp((-(a*(t-tslide(j))).^2)/2);  
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
colormap(hot)
ylim([0 2000])
title(['Messiah spectrogram, Mexican Hat: a = ', num2str(a), ', d\tau = ', num2str(dtau)])
xlabel('time (s)')
ylabel('frequency (Hz)')


% SHANNON WINDOW
a = 0.1; dtau = 0.1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    % Shannon filter function
    g = heaviside((t - tslide(j)) + a) - heaviside((t - tslide(j)) - a);
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
colormap(hot)
ylim([0 4000])
title(['Messiah spectrogram, Shannon Hat: a = ', num2str(a), ', d\tau = ', num2str(dtau)])
xlabel('time (s)')
ylabel('frequency (Hz)')


%% PART II: Music Scores for 'Mary had a Little Lamb'

% PIANO RECORDING
% defining the domain
[S,Fs] = audioread('music1.wav');          
S = S';
L = length(S)/Fs;
n = length(S);
t = linspace(0,L,n);
k = (1/L)*[0:(n/2-1) -n/2:-1];
ks = fftshift(k);

% Gabor filtering
a = 100; dtau = 0.1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    g = exp(-a*(t - tslide(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

% plotting the piano spectrogram
cutoff = 417301; % where ks frequency = 4186 Hz (arbitrary limit defined by highest piano key)
ks = ks(end/2 + 1:cutoff);
Sgt_spec = Sgt_spec(:, end/2 + 1:cutoff);
pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
ylim([0 1000])
colormap(hot)
title(['Piano Spectrogram, a = ', num2str(a)])
xlabel('time (s)')
ylabel('frequency (Hz)')


% RECORDER RECORDING
% defining the domain
[S,Fs] = audioread('music2.wav');          % import audio file
S = S';
L = length(S)/Fs;
n = length(S);
t = linspace(0,L,n);
k = (1/L)*[0:(n/2-1) -n/2:-1]; 
ks = fftshift(k);

% Gabor filtering
a = 5000; dtau = 0.1;
tslide = 0:dtau:L;
Sgt_spec = zeros(length(tslide),n);
for j = 1:length(tslide)
    g = exp(-a*(t - tslide(j)).^2);
    Sg = g.*S;
    Sgt = fft(Sg); 
    Sgt_spec(j,:) = fftshift(abs(Sgt));
end

% plotting the recorder spectrogram
cutoff = 373438; % where ks frequency = 4186 Hz (arbitrary limit defined by highest piano key)
ks = ks(end/2 + 1:cutoff);
Sgt_spec = Sgt_spec(:, end/2 + 1:cutoff);
pcolor(tslide,ks,Sgt_spec.'), 
shading interp 
ylim([0 4186])
colormap(hot)
title(['Recorder Spectrogram, a = ', num2str(a)])
xlabel('time (s)')
ylabel('frequency (Hz)')