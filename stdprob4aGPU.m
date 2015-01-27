nx = 166; % number of cells on x direction
ny = 42;
nz = 1;
dd = 3; % cell volume = dd x dd x dd
dt = 5E-6; % timestep in nanoseconds
dt = gpuArray(dt); % copy data from CPU to GPU
timesteps = 150000;
alpha = 0.5; % damping constant to relax system to S-state
alpha = gpuArray(alpha);
exchConstant = 1.3E-11 * 1E18; % nanometer/nanosecond units

mu_0 = 1.256636; % vacuum permeability, = 4 * pi / 10
Ms = 800; % saturation magnetization
Ms = gpuArray(Ms);
exch = 2 * exchConstant / mu_0 / Ms / Ms;
exch = gpuArray(exch);
prefactor1 = (-0.221) * dt / (1 + alpha * alpha);
prefactor2 = prefactor1 * alpha / Ms;
prefactor1 = gpuArray(prefactor1);
prefactor2 = gpuArray(prefactor2);

Mx = repmat(Ms, [nx ny nz]); % magnetization on x direction 
Mx = gpuArray(Mx);
My = gpuArray.zeros([nx ny nz]); % TODO: reduce truncation and zeropadding into one step
Mz = My;

deltaMx = gpuArray.zeros([nx ny nz]);
deltaMy = gpuArray.zeros([nx ny nz]);
deltaMz = gpuArray.zeros([nx ny nz]);
mag = gpuArray.zeros([nx ny nz]);

Kxx = zeros(nx * 2, ny * 2, nz * 2); % Initialization of demagnetization tensor
Kxy = Kxx;
Kxz = Kxx;
Kyy = Kxx;
Kyz = Kxx;
Kzz = Kxx;
prefactor = 1 / 4 / 3.14159265;
for K = -nz + 1 : nz - 1 % Calculation of Demag tensor
    for J = -ny + 1 : ny - 1
        for I = -nx + 1 : nx - 1
            if I == 0 && J == 0 && K == 0
                continue
            end
            L = I + nx; % shift the indices, b/c no negative index allowed in MATLAB
            M = J + ny;
            N = K + nz;
            for i = 0 : 1 % helper indices
                for j = 0 : 1
                    for k = 0 : 1
                        r = sqrt ( (I+i-0.5)*(I+i-0.5)*dd*dd +(J+j-0.5)*(J+j-0.5)*dd*dd +(K+k-0.5)*(K+k-0.5)*dd*dd);                        
                        Kxx(L,M,N) = Kxx(L,M,N) + (-1).^(i+j+k) * atan( (K+k-0.5) * (J+j-0.5) * dd / r / (I+i-0.5));                        
                        Kxy(L,M,N) = Kxy(L,M,N) + (-1).^(i+j+k) * log( (K+k-0.5) * dd + r);                        
                        Kxz(L,M,N) = Kxz(L,M,N) + (-1).^(i+j+k) * log( (J+j-0.5) * dd + r);                        
                        Kyy(L,M,N) = Kyy(L,M,N) + (-1).^(i+j+k) * atan( (I+i-0.5) * (K+k-0.5) * dd / r / (J+j-0.5));                        
                        Kyz(L,M,N) = Kyz(L,M,N) + (-1).^(i+j+k) * log( (I+i-0.5) * dd + r);                        
                        Kzz(L,M,N) = Kzz(L,M,N) + (-1).^(i+j+k) * atan( (J+j-0.5) * (I+i-0.5) * dd / r / (K+k-0.5));
                    end
                end
            end
            Kxx(L,M,N) = Kxx(L,M,N) * prefactor;
            Kxy(L,M,N) = Kxy(L,M,N) * - prefactor;
            Kxz(L,M,N) = Kxz(L,M,N) * - prefactor;
            Kyy(L,M,N) = Kyy(L,M,N) * prefactor;
            Kyz(L,M,N) = Kyz(L,M,N) * - prefactor;
            Kzz(L,M,N) = Kzz(L,M,N) * prefactor;
        end
    end
end % calculation of demag tensor done

Kxx_fft = fftn(Kxx); % fast fourier transform of demag tensor
Kxy_fft = fftn(Kxy); % need to be done only one time
Kxz_fft = fftn(Kxz);
Kyy_fft = fftn(Kyy);
Kyz_fft = fftn(Kyz);
Kzz_fft = fftn(Kzz);

Kxx_fft = gpuArray(Kxx_fft);
Kxy_fft = gpuArray(Kxy_fft);
Kxz_fft = gpuArray(Kxz_fft);
Kyy_fft = gpuArray(Kyy_fft);
Kyz_fft = gpuArray(Kyz_fft);
Kzz_fft = gpuArray(Kzz_fft);

Hx_exch = gpuArray.zeros(nx, ny, nz);
Hy_exch = gpuArray.zeros(nx, ny, nz);
Hz_exch = gpuArray.zeros(nx, ny, nz);

outFile = fopen('Mdata.txt', 'w');

Hx0 = gpuArray.zeros(nx, ny, nz);
Hx1 = gpuArray.zeros(nx, ny, nz);
Hx2 = gpuArray.zeros(nx, ny, nz);
Hx3 = gpuArray.zeros(nx, ny, nz);

Hy0 = gpuArray.zeros(nx, ny, nz);
Hy1 = gpuArray.zeros(nx, ny, nz);
Hy2 = gpuArray.zeros(nx, ny, nz);
Hy3 = gpuArray.zeros(nx, ny, nz);

Hz0 = gpuArray.zeros(nx, ny, nz);
Hz1 = gpuArray.zeros(nx, ny, nz);
Hz2 = gpuArray.zeros(nx, ny, nz);
Hz3 = gpuArray.zeros(nx, ny, nz);

for t = 1 : timesteps
    Mx(end + nx, end + ny, end + nz) = 0; % zero padding
    My(end + nx, end + ny, end + nz) = 0;
    Mz(end + nx, end + ny, end + nz) = 0;
    
    Hx = ifftn(fftn(Mx) .* Kxx_fft + fftn(My) .* Kxy_fft + fftn(Mz) .* Kxz_fft); % calc demag field with fft
    Hy = ifftn(fftn(Mx) .* Kxy_fft + fftn(My) .* Kyy_fft + fftn(Mz) .* Kyz_fft);
    Hz = ifftn(fftn(Mx) .* Kxz_fft + fftn(My) .* Kyz_fft + fftn(Mz) .* Kzz_fft);

    Hx = real(Hx (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) ) ); % truncation of demag field
    Hy = real(Hy (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) ) );
    Hz = real(Hz (nx:(2 * nx - 1), ny:(2 * ny - 1), nz:(2 * nz - 1) ) );
    Mx = Mx (1:nx, 1:ny, 1:nz); % truncation of Mx, remove zero padding
    My = My (1:nx, 1:ny, 1:nz);
    Mz = Mz (1:nx, 1:ny, 1:nz);
    % calculation of exchange field
    Hx0 (2:end,:,:) = Mx(1:end-1,:,:); % -x
    Hx0 (1,:,:) = Hx0(2,:,:);
    Hx1 (1:end-1,:,:) = Mx(2:end,:,:); % +x
    Hx1 (end,:,:) = Hx1(end-1,:,:);
    
    Hx2 (:,2:end,:) = Mx(:,1:end-1,:); % -y
    Hx2 (:,1,:) = Hx2(:,2,:);
    Hx3 (:,1:end-1,:) = Mx(:,2:end,:); % +y
    Hx3 (:,end,:) = Hx3(:,end-1,:);
    
    Hy0 (2:end,:,:) = My(1:end-1,:,:);
    Hy0 (1,:,:) = Hy0(2,:,:);
    Hy1 (1:end-1,:,:) = My(2:end,:,:);
    Hy1 (end,:,:) = Hy1(end-1,:,:);
    
    Hy2 (:,2:end,:) = My(:,1:end-1,:);
    Hy2 (:,1,:) = Hy2(:,2,:);
    Hy3 (:,1:end-1,:) = My(:,2:end,:);
    Hy3 (:,end,:) = Hy3(:,end-1,:);
    
    Hz0 (2:end,:,:) = Mz(1:end-1,:,:);
    Hz0 (1,:,:) = Hz0(2,:,:);
    Hz1 (1:end-1,:,:) = Mz(2:end,:,:);
    Hz1 (end,:,:) = Hz1(end-1,:,:);
    
    Hz2 (:,2:end,:) = Mz(:,1:end-1,:);
    Hz2 (:,1,:) = Hz2(:,2,:);
    Hz3 (:,1:end-1,:) = Mz(:,2:end,:);
    Hz3 (:,end,:) = Hz3(:,end-1,:);

    Hx = Hx + exch / dd / dd * (Hx0 + Hx1 + Hx2 + Hx3 - 4 * Mx);
    Hy = Hy + exch / dd / dd * (Hy0 + Hy1 + Hy2 + Hy3 - 4 * My);
    Hz = Hz + exch / dd / dd * (Hz0 + Hz1 + Hz2 + Hz3 - 4 * Mz);

    if t < 4000
        Hx = Hx + 100; % apply a saturation field to get S-state
        Hy = Hy + 100;
        Hz = Hz + 100;
    elseif t < 6000 
        Hx = Hx + (6000 - t) / 20; % gradually diminish the field
        Hx = Hx + (6000 - t) / 20;
        Hx = Hx + (6000 - t) / 20;
    elseif t > 50000
        Hx = Hx - 19.576; % apply the reverse field
        Hy = Hy + 3.422;
        alpha = 0.02;
        prefactor1 = (-0.221) * dt / (1 + alpha * alpha);
        prefactor2 = prefactor1 * alpha / Ms;
    end
    % apply LLG equation
    MxHx = My .* Hz - Mz .* Hy; % = M cross H
    MxHy = Mz .* Hx - Mx .* Hz;
    MxHz = Mx .* Hy - My .* Hx;    
    
    deltaMx = prefactor1 .* MxHx + prefactor2 .* (My .* MxHz - Mz .* MxHy); 
    deltaMy = prefactor1 .* MxHy + prefactor2 .* (Mz .* MxHx - Mx .* MxHz);
    deltaMz = prefactor1 .* MxHz + prefactor2 .* (Mx .* MxHy - My .* MxHx);
    
    Mx = Mx + deltaMx;
    My = My + deltaMy;
    Mz = Mz + deltaMz;
    
    mag = sqrt(Mx .* Mx + My .* My + Mz .* Mz);
    Mx = Mx ./ mag * Ms;
    My = My ./ mag * Ms;
    Mz = Mz ./ mag * Ms;
    
    if mod(t, 1000) == 0 % recod the average magnetization
        MxMean = mean(Mx(:));
        MyMean = mean(My(:));
        MzMean = mean(Mz(:));
        fprintf(outFile, '%d\t%f\t%f\t%f\r\n', t, MxMean/Ms, MyMean/Ms, MzMean/Ms);
    end
end
fclose('all');


