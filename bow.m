function descrs = bow(imPath)
    samplesize = 50;
    vocabsize = 400;
    numdescr = 1000; %number of descriptors
    disp('Loading images')
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
    motors_valid = load_images(imPath, samplesize);
    imPath = 'CalTech4/imageData/faces_test';
    faces_valid = load_images(imPath, samplesize); 
    imPath = 'CalTech4/imageData/airplanes_test';
    airplanes_valid = load_images(imPath, samplesize); 
    imPath = 'CalTech4/imageData/cars_test';
    cars_valid = load_images(imPath, samplesize); 
    
    all_images_valid = [cars_valid, airplanes_valid, faces_valid, motors_valid]; 
    disp('Getting descriptors')
    [xs, ys, ids] = get_features(all_images);
    [xs_valid, ys_valid, ids_valid] = get_features(all_images_valid);
    %cluster data
    disp('Clustering data')
    % data = double(vl_colsubset(descs, numdescr));
    xs = reshape(xs,1,[]);
    ys = reshape(ys,1,[]);
    data = [xs ; ys];
    xs_valid = reshape(xs_valid,1,[]);
    ys_valid = reshape(ys_valid,1,[]);
    data_valid = [xs_valid ; ys_valid];
    % data = data(:,randperm(size(data,2)));
    % data = data(:,1:numdescr);
    disp(size(data));
% find clusters
    disp('k-means')
    [centers, assignments] = vl_kmeans(data, vocabsize,'distance', 'l2', 'algorithm', 'ELKAN');  
    tree = [];
    %vocabsize = vocabsize
    tree.K = vocabsize;
    tree.depth = 1;
    tree.centers = int32(centers);
    %disp(tree)
    cc=hsv(vocabsize);
    figure;
    hold on;
    plot(centers(1,:),centers(2,:),'k.','MarkerSize',20);
    for i=1:vocabsize
        plot(data(1,assignments == i),data(2,assignments == i),'.','color',cc(i,:));
    end
  
%visual words
    % disp('Quantizing');
    % p = vl_hikmeanspush( tree, assignments(ids==1) );   
%histogram  
    % h = vl_hikmeanshist(tree,p);
    % disp(h)
    
    
    for j=1:4
        disp('Processing data');
        train_data = [];
        train_labels = [];
        valid_data = [];
        valid_labels = [];
       for k=1:4
         for i=1:10
            h = histogram(assignments(ids==(k-1)*samplesize+i),vocabsize);
            features = h.Values;
            features = 1.0 * features ./ max(features);
            features = reshape(features, 1, []);
            train_data = [train_data; features];
            if k == j
                train_labels = [train_labels; 1];
            else
                train_labels = [train_labels; 0];
            end
         end
         for i=11:20
            h = histogram(assignments(ids==(k-1)*samplesize+i),vocabsize);
            features = h.Values;
            features = 1.0 * features ./ max(features);
            features = reshape(features, 1, []);
            valid_data = [valid_data; features];
            if k == j
                valid_labels = [valid_labels; 1];
            else
                valid_labels = [valid_labels; 0];
            end
         end
       end
       size(train_data)
       size(valid_data)
        disp('Training');
         model = svmtrain(train_labels, train_data, '-g 0.07 -b 1');
        [prediction, accuracy, prob_values] = svmpredict(valid_labels, valid_data, model, '-b 1');
    end
    
end

function [xs,ys, ids] = get_features(arrImages)
    xs = [];
    ys = [];
    ids = [];
    for i=1:size(arrImages, 2)
        
        im = fitimage(arrImages{i});
        if size(im,3) == 3
            im = vl_imsmooth(im2single(rgb2gray(im)),2);
        else
            im = vl_imsmooth(im2single(im),2);
        end
        [frames, ~]= vl_sift(im) ;
        new_xs = frames(1,:);
        is =  repelem([i], size(new_xs,2));
        xs = [xs new_xs];
        ys = [ys frames(2,:)];
        ids = [ids is];
    end   
end

function arrImages = load_images(imPath, samplesize)
    for i=1:samplesize
        
        imName = strcat('img', sprintf('%03d',i), '.jpg');
        imName = fullfile(imPath, imName) ;
        arrImages{i}=imread(imName);
        imName ='';
    end
end

function im = fitimage(im)
% -------------------------------------------------------------------------

    im = im2single(im) ;
    if size(im,1) > 480 
        im = imresize(im, [480 NaN]) ; 
    end    
end
