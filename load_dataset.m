function [all_images, all_images_valid] = load_dataset(samplesize)
imPath = 'CalTech4/imageData/motorbikes_train';
motors = load_images(imPath, samplesize);
imPath = 'CalTech4/imageData/faces_train';
faces = load_images(imPath, samplesize);
imPath = 'CalTech4/imageData/airplanes_train';
airplanes = load_images(imPath, samplesize);
imPath = 'CalTech4/imageData/cars_train';
cars = load_images(imPath, samplesize);

all_images = [cars, airplanes, faces, motors];

imPath = 'CalTech4/imageData/motorbikes_test';
motors_valid = load_images(imPath, 50);
imPath = 'CalTech4/imageData/faces_test';
faces_valid = load_images(imPath, 50);
imPath = 'CalTech4/imageData/airplanes_test';
airplanes_valid = load_images(imPath, 50);
imPath = 'CalTech4/imageData/cars_test';
cars_valid = load_images(imPath, 50);

all_images_valid = [cars_valid, airplanes_valid, faces_valid, motors_valid];
end

function arrImages = load_images(imPath, samplesize)
for i=1:samplesize
    
    imName = strcat('img', sprintf('%03d',i), '.jpg');
    imName = fullfile(imPath, imName) ;
    arrImages{i}=imread(imName);
    imName ='';
end
end
