clear all; close all; clc;
load HW1_Testdata.mat

%%% Name: ROSEMICHELLE MARZAN
%%% Course: AMATH 482
%%% Homework 1, Due 1/24/2019

% VARIABLE NAMES FOR SIGNAL
% Undata = raw (noisy) data
% Unt = noisy data in the frequency domain
% Unft = filtered data in frequency domain
% Unf = filtered data in time domain

% setting up the domains
L = 15; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; 
[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(k,k,k);
samples = 20; % num of measurements

% averaging the spectrum
ave = zeros(n,n,n);
for i = 1:samples
    ave = ave + fftn(reshape(Undata(i,:),n,n,n));
end
ave = abs(ave)/samples;

% find the marble's frequency signature
[val,index] = max(ave(:));
[xind,yind,zind] = ind2sub([n,n,n],index);
K0x = Kx(xind,yind,zind);
K0y = Ky(xind,yind,zind);
K0z = Kz(xind,yind,zind);

% filtering noisy data at the center frequency
tau = 0.2;
filter = exp(-tau.*((Kx-K0x).^2 + (Ky-K0y).^2 + (Kz-K0z).^2));
% marble location as (x,y,z) coordinate
marbleloc = zeros(samples,3);

for i = 1:samples
    Unt = fftn(reshape(Undata(i,:),[n,n,n]));
    Unft = Unt.*filter;
    Unf = ifftn(Unft);
    [val,index] = max(Unf(:));
    [xind, yind, zind] = ind2sub([n,n,n],index);
    marbleloc(i,1) = X(xind,yind,zind); % x-coordinate
    marbleloc(i,2) = Y(xind,yind,zind); % y-coordinate
    marbleloc(i,3) = Z(xind,yind,zind); % z-coordinate
end

plot3(marbleloc(:,1),marbleloc(:,2),marbleloc(:,3),'-o','Linewidth',2);
title('Path of marble over time')
xlabel('x');
ylabel('y');
zlabel('z');
axis([-15 15 -15 15 -15 15])
hold on
plot3(marbleloc(end,1),marbleloc(end,2),marbleloc(end,3),'ro','MarkerFaceColor','r','Linewidth',2);

% marble location at 20th measurement
final = marbleloc(end,:);
