% SIM psf calculation for hexSIM with light sheet
% Simulates the raw images produced while scanning an object (3d point 
% cloud) through focus as the hexSIM illumination pattern is shifted 
% through 7 positions laterally.
% Finally it processes the illumination using the hxSimProcessor class to 
% generate superresolved output.

N=256;          % Points to use in FFT
pixelsize = 3.25;    % Camera pixel size
magnification = 60; % Objective magnification
dx=pixelsize/magnification;     % Sampling in lateral plane at the sample in um
NA=1.1;         % Numerical aperture at sample
n=1.33;         % Refractive index at sample
lambda=0.525;   % Wavelength in um
npoints=1000;   % Number of random points
rng(1234);      % set random number generator seed
% eta is the factor by which the illumination grid frequency exceeds the
% incoherent cutoff, eta=1 for normal SIM, eta=sqrt(3)/2 to maximise
% resolution without zeros in TF
% carrier is 2*kmax*eta

import_data = true;    % Whether to import a raw image data, or to simulate a new one

eta=1.0;
axial=false;     % Whether to model axial or in-plane polarised illumination

dxn = lambda/(4*NA);          % 2*Nyquist frequency in x and y.
Nn = ceil(N*dx/dxn/2)*2;      % Number of points at Nyquist sampling, even number
dxn = N*dx/Nn;                % correct spacing
res = lambda/(2*NA);
oversampling = res/dxn;       % factor by which pupil plane oversamples the coherent psf data

dk=oversampling/(Nn/2);       % Pupil plane sampling
[kx,ky] = meshgrid(-dk*Nn/2:dk:dk*Nn/2-dk,-dk*Nn/2:dk:dk*Nn/2-dk);

kr=sqrt(kx.^2+ky.^2);

% Raw pupil function, pupil defined over circle of radius 1.
csum=sum(sum((kr<1))); % normalise by csum so peak intensity is 1

zrange=7;          % distance either side of focus to calculate
alpha=asin(NA/n);
dzn=0.8*lambda/(2*n*(1-cos(alpha)));    % Nyquist sampling in z, reduce by 10% to account for gaussian light sheet
dz=0.4;             % step size in axial direction of PSF
Nz=2*ceil(zrange/dz);
dz=2*zrange/Nz;
Nzn=2*ceil(zrange/dzn);
dzn=2*zrange/Nzn;
if Nz < Nzn
    Nz = Nzn;
    dz = dzn;
end

%% Import or simulate raw image data
if import_data
    load('img_256_axial_x2_1000points.mat');
else
    % Calculate 3d PSF
    clear psf;
    psf=zeros(Nn,Nn,Nzn);
    c=zeros(Nn);
    fwhmz=3;            % FWHM of light sheet in z
    sigmaz=fwhmz/2.355;

    pupil = (kr<1);
    nz = 1;
    disp("Calculating 3d psf");

    tic
    
    for z = -zrange:dzn:zrange-dzn
        c(pupil) = exp(1i*(z*n*2*pi/lambda*sqrt((1-kr(pupil).^2*NA^2/n^2))));
        psf(:,:,nz) = abs(fftshift(ifft2(c))).^2*exp(-z^2/2/sigmaz^2);
        nz = nz+1; 
    end

    % Normalised so power in resampled psf (see later on) is unity in focal plane
    psf = psf * Nn^2/sum(pupil(:))*Nz/Nzn; 

    toc

    % Calculate 3D-OTF
    disp("Calculating 3d otf");

    tic
    otf = fftn(psf);
    toc

    aotf = abs(fftshift(otf));
    m = max(aotf(:));

    % Set up some random points
    disp("Calculating point cloud");
    tic
    rad = 5;        % radius of sphere of points
    pointsx2 = (2*rand(npoints*2,3)-1).*[rad,rad,rad];

    pointsx2r = sum(pointsx2.*pointsx2,2);

    points_sphere = pointsx2(pointsx2r<rad^2,:);

    points = points_sphere((1:npoints),:);
%     points(:,3)=points(:,3)/2;
    toc

    % Generate phase tilts in frequency space

    xyrange = Nn/2*dxn;
    dkxy = pi/xyrange;
    kxy = -Nn/2*dkxy:dkxy:(Nn/2-1)*dkxy;
    dkz = pi/zrange;
    kz = -Nzn/2*dkz:dkz:(Nzn/2-1)*dkz;

    phasetilts=complex(single(zeros(Nn,Nn,Nzn,7)));

    disp("Calculating pointwise phase tilts");

    tic

    for j = 1:7
        pxyz = complex(single(zeros(Nn,Nn,Nzn)));
        for i = 1:npoints
            x=points(i,1);
            y=points(i,2);
            z=points(i,3)+dz/7*(j-1); 
            ph=eta*4*pi*NA/lambda;
            p1=-j*2*pi/7;
            p2=j*4*pi/7;
            if axial % axial polarisation normalised to peak intensity of 1
                ill = 2/9*(3/2+cos(ph*(y)+p1-p2)...
                    +cos(ph*(y-sqrt(3)*x)/2+p1)...
                    +cos(ph*(-y-sqrt(3)*x)/2+p2));
            else     % in plane polarisation normalised to peak intensity of 1
                ill = 2/9*(3-cos(ph*(y)+p1-p2)...
                    -cos(ph*(y-sqrt(3)*x)/2+p1)...
                    -cos(ph*(-y-sqrt(3)*x)/2+p2));
            end
            px = exp(1i*single(x*kxy));
            py = exp(1i*single(y*kxy));
            pz = exp(1i*single(z*kz))*ill;
            pxy = px.'*py;
            for ii = 1:length(kz)
                pxyz(:,:,ii) = pxy.*pz(ii);
            end
            phasetilts(:,:,:,j) = phasetilts(:,:,:,j)+pxyz;
        end
    end
    toc

    % calculate output

    disp("Calculating raw image stack");

    tic

    img = zeros(N,N,Nz*7,'single');

    for j = 1:7
        ootf = fftshift(otf) .* phasetilts(:,:,:,j);
        img(:,:,j:7:end) = abs(ifftn(ootf,[N N Nz])); 
                % OK to use abs here as signal should be all positive.
                % Abs is required as the result will be complex as the 
                % fourier plane cannot be shifted back to zero when oversampling.
                % But should reduction in sampling be allowed here (Nz<Nzn)?
    end
    toc

end

%% Calibration

h=hexSimProcessor();
h.NA=NA;
h.pixelsize=pixelsize;
h.magnification=magnification;
h.w=0.05;
h.beta=0.99;
h.cleanup=false;
h.eta=0.8*eta;
h.axial=axial;
h.N=N;
h.debug=false;
h.usemodulation=true;

disp("Calibration");
tic
% profile on
h.calibrate(img(:,:,7*Nz/2+1:7*Nz/2+7)+img(:,:,7*Nz/2-13:7*Nz/2-7)+img(:,:,7*Nz/2+15:7*Nz/2+21));
% profile off
% profile viewer
h.reset()
toc

%% Reconstruct raw data (conventional image)

disp("Calculating conventional image stack");
tic
fs=fft(img,7*Nz,3);
imgz=ifft(fs(:,:,[1:Nz/2, end-(Nz/2-1):end]),Nz,3,'symmetric');
toc
clear fs;

%% Reconstruct super-resolved data

disp("Calculating z-packed image stack in compat batch mode");
tic
imgout = h.batchreconstructcompact(img);
toc

%% Set up parameters for 3D flythrough

facecolor = [0.9 0.9 0.9];
frame_num = 180;
zoom = 1;

% orbit_hor = 180;
% orbit_ver = 90;

orbit_hor_array = 90/frame_num * ones(1,frame_num);

% orbit_hor_array = sin((1:frame_num)/frame_num*pi/2);
% orbit_hor_array = orbit_hor_array/sum(orbit_hor_array) * 180;

orbit_ver_array = cos((1:frame_num)/frame_num*pi/2);
orbit_ver_array = orbit_ver_array/sum(orbit_ver_array) * 90;


%% Create video for the raw data, with iso value determined by histogram
Imax_in=max(imgz(:));

Itot_in=imgz/Imax_in;

isovalue_in = fun_find_isovalue(Itot_in);

[xg_in, yg_in, zg_in] = meshgrid(-N/2*dx:dx:(N/2-1)*dx,-N/2*dx:dx:(N/2-1)*dx,-Nz/2*dz:dz:(Nz/2-1)*dz);

fun_flythrough_3D(xg_in,yg_in,zg_in,Itot_in,isovalue_in,facecolor,frame_num,zoom,orbit_hor_array,orbit_ver_array,'Video_input_from_fun');

%% Create video for the reconstructed data, with iso value determined by histogram
Imax_out=max(imgout(:));

Itot_out=imgout/Imax_out;

isovalue_out = fun_find_isovalue(Itot_out);

[xg_out, yg_out, zg_out] = meshgrid(-N*dx/2:dx/2:(N-1)*dx/2,-N*dx/2:dx/2:(N-1)*dx/2,-Nz/2*dz:dz:(Nz/2-1)*dz);

fun_flythrough_3D(xg_out,yg_out,zg_out,Itot_out,isovalue_out,facecolor,frame_num,zoom,orbit_hor_array,orbit_ver_array,'Video_output_from_fun');

%% User-defined functions in use

function fun_flythrough_3D(X, Y, Z, img, iso_value, facecolor, frame_num, zoom_value, orbit_hor_array, orbit_ver_array, vidfile_name)

    f = figure();

    clf();

    % whitebg('black');
    set(gca,'Color','black');
    set(gcf,'Color',[0.1,0.1,0.1])

    set(f,'renderer','opengl')

    p=patch(isosurface(X,Y,Z,img,iso_value),'FaceColor',facecolor,'EdgeColor','none');
    isonormals(X,Y,Z,img,p);
    daspect([1 1 1])
    view(0,0); 
    axis tight
    cam = camlight('right');
    lighting phong

    % Initialize video
    myVideo = VideoWriter(vidfile_name,'MPEG-4'); %open video file
    myVideo.FrameRate = frame_num/5;  %can adjust this
    open(myVideo);

    for idx = 1:frame_num
        % Update current view.
%         camorbit(orbit_hor/frame_num,orbit_ver/frame_num);
        camorbit(orbit_hor_array(idx),orbit_ver_array(idx));
        camzoom(nthroot(zoom_value,frame_num));
        camlight(cam,'right');

        drawnow;

    %     pause(1/myVideo.FrameRate);
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end
    
    for idx2 = frame_num:-1:1
%         camorbit(orbit_hor/frame_num,-orbit_ver/frame_num);
        camorbit(orbit_hor_array(idx2),-orbit_ver_array(idx2));
        camzoom(nthroot(1/zoom_value,frame_num));
        camlight(cam,'right');

        drawnow;

    %     pause(1/myVideo.FrameRate);
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end
    
    for idx3 = 1:frame_num
%         camorbit(orbit_hor/frame_num,-orbit_ver/frame_num);
        camorbit(orbit_hor_array(idx3),-orbit_ver_array(idx3));
        camzoom(nthroot(1/zoom_value,frame_num));
        camlight(cam,'right');

        drawnow;

    %     pause(1/myVideo.FrameRate);
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end  
    
    for idx4 = frame_num:-1:1
%         camorbit(orbit_hor/frame_num,-orbit_ver/frame_num);
        camorbit(orbit_hor_array(idx4),orbit_ver_array(idx4));
        camzoom(nthroot(1/zoom_value,frame_num));
        camlight(cam,'right');

        drawnow;

    %     pause(1/myVideo.FrameRate);
        frame = getframe(gcf); %get frame
        writeVideo(myVideo, frame);
    end  

    close(myVideo);

end

function isovalue = fun_find_isovalue(img)

    index_img_max_3d = imregionalmax(img);
    
    index_img_max_3d( img<max(img(:))/3 ) = 0;  % May need to come up with a better way for this lower filtering step
    
    img_max_3d = nonzeros(img(index_img_max_3d));
    
    img_max_3d = sort(img_max_3d(:),'ascend');
    
    img_max_3d = img_max_3d(1:round(0.9*length(img_max_3d)));
    
    isovalue = 0.5*median(img_max_3d);
    
end

