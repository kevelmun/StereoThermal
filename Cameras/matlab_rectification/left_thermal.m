% Auto-generated by stereoCalibrator app on 25-Nov-2024
%-------------------------------------------------------


% Define images to process
imageFileNames1 = {'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141752.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141756.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141759.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141804.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141806.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141809.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141812.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141814.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141817.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141820.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141823.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141828.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141832.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141835.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141840.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141843.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141846.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141849.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141852.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141855.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141858.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141901.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141905.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141912.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141915.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141918.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141924.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141929.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141931.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141934.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141937.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141940.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141944.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141950.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_141957.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142000.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142007.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142010.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142014.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142019.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142022.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142028.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142031.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142038.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142040.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142042.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142044.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142046.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142049.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142052.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\left\LEFT_visible_20241113_142055.png',...
    };
imageFileNames2 = {'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141752.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141756.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141759.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141804.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141806.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141809.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141812.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141814.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141817.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141820.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141823.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141828.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141832.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141835.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141840.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141843.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141846.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141849.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141852.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141855.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141858.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141901.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141905.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141912.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141915.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141918.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141924.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141929.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141931.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141934.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141937.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141940.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141944.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141950.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_141957.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142000.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142007.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142010.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142014.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142019.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142022.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142028.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142031.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142038.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142040.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142042.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142044.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142046.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142049.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142052.png',...
    'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\CalibrationData\Steven\thermal_invert\thermal_20241113_142055.png',...
    };

% Detect calibration pattern in images
detector = vision.calibration.stereo.CheckerboardDetector();
minCornerMetric = 0.150000;
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames1, imageFileNames2, 'MinCornerMetric', minCornerMetric);

% Generate world coordinates for the planar pattern keypoints
squareSize = 20.000000;  % in millimeters
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Read one of the images from the first stereo pair
I1 = imread(imageFileNames1{1});
[mrows, ncols, ~] = size(I1);

% Calibrate the camera
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(stereoParams);

% Visualize pattern locations
h2=figure; showExtrinsics(stereoParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, stereoParams);

% You can use the calibration data to rectify stereo images.
I2 = imread(imageFileNames2{1});
[J1, J2, reprojectionMatrix] = rectifyStereoImages(I1, I2, stereoParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('StereoCalibrationAndSceneReconstructionExample')
% showdemo('DepthEstimationFromStereoVideoExample')
%% Rectify new images

inputDir1 = 'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\StereoThermal\Cameras\captures\visible\left';  % Carpeta de imágenes izquierda
inputDir2 = 'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\StereoThermal\Cameras\captures\thermal' % Carpeta de imágenes derecha
outputDir1 = 'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\StereoThermal\Cameras\captures\left_thermal_rectified'; % Carpeta de salida izquierda
outputDir2 = 'C:\Users\usuario\Documents\CiDIS\Stereo_Thermal\Code\StereoThermal\Cameras\captures\thermal_rectified'; % Carpeta de salida derecha

% Crear carpetas de salida si no existen
if ~exist(outputDir1, 'dir')
    mkdir(outputDir1);
end
if ~exist(outputDir2, 'dir')
    mkdir(outputDir2);
end

% Obtener lista de imágenes en ambas carpetas
imageFiles1 = dir(fullfile(inputDir1, '*.png')); % Cambiar *.png por el formato correspondiente (e.g., *.jpg)
imageFiles2 = dir(fullfile(inputDir2, '*.png'));

% Verificar que haya la misma cantidad de imágenes
if numel(imageFiles1) ~= numel(imageFiles2)
    error('El número de imágenes en las carpetas izquierda y derecha no coincide.');
end

% Rectificar cada par de imágenes
for i = 1:numel(imageFiles1)
    % Leer imágenes originales
    inputFile1 = fullfile(inputDir1, imageFiles1(i).name);
    inputFile2 = fullfile(inputDir2, imageFiles2(i).name);
    
    I1 = imread(inputFile1);
    I2 = imread(inputFile2);

    % Validar tamaños
    if size(I1, 1) ~= size(I2, 1) || size(I1, 2) ~= size(I2, 2)
        fprintf('Ajustando dimensiones de la imagen %s y %s\n', imageFiles1(i).name, imageFiles2(i).name);
        I2 = imresize(I2, [size(I1, 1), size(I1, 2)]);
    end

    % Validar número de canales (RGB vs escala de grises)
    if size(I1, 3) ~= size(I2, 3)
        fprintf('Ajustando canales de la imagen %s y %s\n', imageFiles1(i).name, imageFiles2(i).name);
        if size(I1, 3) == 1
            I1 = cat(3, I1, I1, I1); % Convertir escala de grises a RGB
        end
        if size(I2, 3) == 1
            I2 = cat(3, I2, I2, I2); % Convertir escala de grises a RGB
        end
    end

    % Validar tipo de datos
    if ~isa(I1, 'uint8')
        I1 = im2uint8(I1);
    end
    if ~isa(I2, 'uint8')
        I2 = im2uint8(I2);
    end

    % Rectificar imágenes
    [rectifiedI1, rectifiedI2] = rectifyStereoImages(I1, I2, stereoParams);
    
    % Ajustar resolución a la original (opcional)
    targetSize = [size(I1, 2), size(I1, 1)]; % Alto x Ancho de la imagen original
    rectifiedI1 = imresize(rectifiedI1, targetSize);
    rectifiedI2 = imresize(rectifiedI2, targetSize);

    % Girar las imágenes rectificadas 90 grados en sentido horario
    rectifiedI1 = imrotate(rectifiedI1, 90); % Sentido horario
    rectifiedI2 = imrotate(rectifiedI2, 90); % Sentido horario
        
    % Construir las rutas de salida
    outputFile1 = fullfile(outputDir1, imageFiles1(i).name); % Imagen izquierda rectificada
    outputFile2 = fullfile(outputDir2, imageFiles2(i).name); % Imagen derecha rectificada
    
    % Guardar imágenes rectificadas
    imwrite(rectifiedI1, outputFile1);
    imwrite(rectifiedI2, outputFile2);
    
    % Mensaje de progreso
    fprintf('Imagen %d/%d procesada: %s y %s\n', i, numel(imageFiles1), imageFiles1(i).name, imageFiles2(i).name);
end

% Mensaje final
disp('Todas las imágenes han sido rectificadas y guardadas.');