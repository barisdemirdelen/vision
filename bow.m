function bow()
cl_rgb = 'rgb';
cl_hsv = 'hsv';
cl_opp = 'opponent';
cl = cl_rgb;
samplesize = 5;
vocabsize = 800;
numdescr = 2500; %number of descriptors
load_from_file = false;
vocabs = [400,800,1600,2000,4000];

for i=1:5
    vocabsize = vocabs(i);
    samplesize = 100;
    disp(vocabsize);
    disp('Loading dataset')
    [all_images, ~] = load_dataset(samplesize);
    disp('Getting descriptors')
    [centers] = get_kmeans(all_images, cl, numdescr, vocabsize, samplesize, load_from_file);
    tree = get_tree(centers, vocabsize);

    samplesize = 50;
    disp('Loading dataset')
    [all_images, all_images_valid] = load_dataset(samplesize);

    train(tree, cl, all_images, samplesize)
    calculate_map(tree, cl, 2, all_images_valid, samplesize)
end
end

function train(tree, cl, all_images, samplesize)
train_data = [];
for j=1:4
    disp('Processing data');
    train_labels = [];
    for k=1:4
        for i=1:samplesize
            if j == 1
                disp((k-1)*samplesize + i)
                h = get_histogram(tree,all_images{1,(k-1)*samplesize + i}, cl);
                features = h;
                features = 1.0 * features ./ max(features);
                features = reshape(features, 1, []);
                train_data = [train_data; features];
            end
            if k == j
                train_labels = [train_labels; 1];
            else
                train_labels = [train_labels; 0];
            end
        end
    end
    disp('Training');
    model = svmtrain(train_labels, train_data, '-g 0.07 -b 1');
    save(strcat('./svm_model', num2str(j),'-',num2str(samplesize),'-', num2str(tree.K), '.mat'), 'model');
end
save(strcat('./train_data', num2str(samplesize),'-', num2str(tree.K),'.mat'), 'train_data');
end

function highest_model = classify(tree, cl, im)
highest_prob = 0;
highest_model = -1;
for j=1:4
    load(strcat('./svm_model', num2str(j),'-',num2str(samplesize),'-', num2str(tree.K), '.mat'), 'model');
    h = get_histogram(tree,im, cl);
    features = h;
    features = 1.0 * features ./ max(features);
    features = reshape(features, 1, []);
    [~, ~, prob_values] = svmpredict(0, features, model, '-b 1');
    if prob_values(1) > highest_prob
        highest_model = j;
        highest_prob = prob_values(1);
    end
end
end

function map = calculate_map(tree, cl, classifier, test_images, samplesize)
load(strcat('./svm_model', num2str(classifier),'-',num2str(samplesize),'-', num2str(tree.K), '.mat'), 'model');
test_labels = [];
test_data = [];
for k=1:4
    for i=1:samplesize
        disp((k-1)*samplesize + i)
        h = get_histogram(tree,test_images{1,(k-1)*samplesize + i}, cl);
        features = h;
        features = 1.0 * features ./ max(features);
        features = reshape(features, 1, []);
        test_data = [test_data; features];
        if k == classifier
            test_labels = [test_labels; 1];
        else
            test_labels = [test_labels; 0];
        end
    end
end
[prediction, accuracy, prob_values] = svmpredict(test_labels, test_data, model, '-b 1');
map = get_map(prob_values(:,2), test_labels, samplesize);
disp(map);
save(strcat('./test_data', num2str(samplesize),'-', num2str(tree.K),'.mat'), 'test_data');
end

function descs = get_descriptors(arrImages, numdescr, cl)
%descs = zeros(384, 72038*size(arrImages, 2));
descs = zeros(384, numdescr*size(arrImages, 2));
%    size(arrImages)
for i=1:size(arrImages, 2)
    
    im = fitimage(arrImages{i});
    [~, d]=vl_phow(im2single(im),'Color', cl) ;
    %size(d)
    k = randperm(size(d,2));
    sample_descs = d(:,k(1:numdescr));
    descs(:,(i-1)*numdescr + 1:i*numdescr) = uint8(sample_descs);
    disp(i)
    
end
end

function hist = get_histogram(tree, im, cl)
im = fitimage(im);
[~, desc]=vl_phow(im2single(im),'Color', cl) ;
desc = uint8(desc);
p = vl_hikmeanspush( tree, desc );
hist = vl_hikmeanshist(tree,p);
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

function map= get_map(prediction_probs, labels, samplesize)
correct = 0;
total = 0;
map = 0;
to_sort = horzcat(prediction_probs, labels);
to_sort = sortrows(to_sort, -1);

prediction = to_sort(:,1);
labels = to_sort(:,2);

for m=1:size(prediction)
    if m > samplesize
        break;
    end
    total = total+1;
    if 1 == labels(m)
        correct = correct + 1;
    end
    map = map + correct / m;
end
map = map / total;
disp(strcat('Mean Average Precision: ', num2str(map)));
end

function tree=get_tree(centers, vocabsize)
tree = [];
tree.K = vocabsize;
tree.depth = 1;
tree.centers = int32(centers);
end

function [centers] = get_kmeans(all_images, cl, numdescr, vocabsize, samplesize, load_from_file)
if load_from_file
    load(strcat('./centers', num2str(samplesize),'-', num2str(vocabsize),'.mat'));
else
    
    [descs] = get_descriptors(all_images,numdescr, cl);
    
    %cluster data
    disp('Clustering data')
    % data = double(vl_colsubset(descs, numdescr));
    % data = data(:,randperm(size(data,2)));
    % data = data(:,1:numdescr);
    % find clusters
    disp('k-means')
    % sample_descs = double(vl_colsubset(descs, numdescr));
    sample_descs = descs;
    [centers, ~] = vl_kmeans(sample_descs, vocabsize,'distance', 'l2', 'algorithm', 'ELKAN');
    
    %disp(tree)
    %visual words
    
    save(strcat('./centers', num2str(samplesize),'-', num2str(vocabsize),'.mat'),'centers');
end
end

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
