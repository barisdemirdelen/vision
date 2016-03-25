function descrs = bow(imPath, cl, mode, im) %mode={TRA,TS,CL}
    cl_rgb = 'rgb';
    cl_hsv = 'hsv';
    cl_opp = 'opponent';
    samplesize = 5;
    vocabsize = 10.0;
    numdescr = 100; %number of descriptors
    %read cars

    %read motors
    imPath = 'CalTech4/imageData/motorbikes_train';
    motors = load_images(imPath, samplesize);
    imPath = 'CalTech4/imageData/faces_train';
    faces = load_images(imPath, samplesize); 
    imPath = 'CalTech4/imageData/airplanes_train';
    airplanes = load_images(imPath, samplesize); 
    imPath = 'CalTech4/imageData/cars_train';
    cars = load_images(imPath, samplesize); 
    
    all_images = [cars, airplanes, faces, motors]; 
    %[~, d] = vl_sift(rgb2gray(im2single(all_images{1})));
    %[f, d] = vl_phow(im2single(all_images{1}),'Color', cl_rgb);
    descs = get_descriptors(all_images, cl_rgb);
    %cluster data
    d = vl_colsubset(descs, numdescr);
% find clusters
    [C, ~] = vl_kmeans(double(d), vocabsize,'distance', 'l2', 'algorithm', 'ELKAN');  
    tree = [];
    %vocabsize = vocabsize
    tree.K = vocabsize;
    tree.depth = 1;
    tree.centers = int32(C);
    %disp(tree)
%visual words
    p = vl_hikmeanspush( tree, descs ); 
    size(p);
    
%histogram    
    h = vl_hikmeanshist(tree,p);

end

function descs = get_descriptors(arrImages, cl)
    %descs = zeros(384, 72038*size(arrImages, 2));
    descs = [];
%    size(arrImages)
    for i=1:size(arrImages, 2)
        
        im = fitimage(arrImages{i});
        [~, d]=vl_phow(im2single(im),'Color', cl) ;
        %size(d)
        
        descs =horzcat(descs, uint8(d));
        %size(descs)
        
    end    
end

function arrImages = load_images(imPath, samplesize)

    for i=1:samplesize
        
        imName = strcat('img', sprintf('%03d',i), '.jpg');
        imName = fullfile(imPath, imName) ;
        arrImages{i}=imread(imName);
        %imshow(cars{i});
        imName ='';
    end
end


function p = vw (tree, descs, k)
    %tree = vl_kdtreebuild(centers); 
    %whos(tree)
    %tree.centers = centers;
    p = vl_hikmeanspush( tree, descs );
end

function hist = getHist(centers, p, k)
    tree = vl_kdtreebuild(centers); 
    hist = vl_hikmeanshist(tree,p);

end

function im = fitimage(im)
% -------------------------------------------------------------------------

    im = im2single(im) ;
    if size(im,1) > 480 
        im = imresize(im, [480 NaN]) ; 
    end    
end
