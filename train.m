function train()
cl_rgb = 'rgb';
cl_hsv = 'hsv';
cl_opp = 'opponent';
cl = cl_opp;
samplesize = 5;
vocabsize = 800;
numdescr = 2500; %number of descriptors
load_from_file = false;
vocabs = [400,800,1200];

for j=1:3
    if j == 1
        cl = cl_opp;
    end
    if j == 2
        cl = cl_hsv;
    end
    if j == 3
        cl = cl_rgb;
    end
    for i=1:3
        vocabsize = vocabs(i);
        samplesize = 10;
        disp(vocabsize);
        disp('Loading dataset')
        [all_images, ~] = load_dataset(samplesize);
        disp('Getting descriptors')
        [centers] = get_kmeans(all_images, cl, numdescr, vocabsize, samplesize, load_from_file);
        
        samplesize = 50;
        disp('Loading dataset')
        [all_images, all_images_valid] = load_dataset(samplesize);
        
        train_models(centers, cl, all_images, samplesize)
        
        % calculate a map to generate test histograms
        calculate_map(centers, cl, 2, all_images_valid, samplesize)
    end
end
end

function train_models(centers, cl, all_images, samplesize)
train_data = [];
for j=1:4
    disp('Processing data');
    train_labels = [];
    for k=1:4
        for i=1:samplesize
            if j == 1
                disp((k-1)*samplesize + i)
                h = get_histogram(centers,all_images{1,(k-1)*samplesize + i}, cl);
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
    save(strcat('models/svm_model',cl, num2str(j),'-',num2str(samplesize),'-', num2str(size(centers,2)), '.mat'), 'model');
end
save(strcat('models/train_data',cl, num2str(samplesize),'-', num2str(size(centers,2)),'.mat'), 'train_data');
end

function highest_model = classify(centers, cl, im)
highest_prob = 0;
highest_model = -1;
for j=1:4
    load(strcat('models/svm_model',cl, num2str(j),'-',num2str(samplesize),'-', num2str(size(centers,2)), '.mat'), 'model');
    h = get_histogram(centers,im, cl);
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

function map = calculate_map(centers, cl, classifier, test_images, samplesize)
load(strcat('models/svm_model',cl, num2str(classifier),'-',num2str(samplesize),'-', num2str(size(centers,2)), '.mat'), 'model');
test_labels = [];
test_data = [];
for k=1:4
    for i=1:samplesize
        disp((k-1)*samplesize + i)
        h = get_histogram(centers,test_images{1,(k-1)*samplesize + i}, cl);
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
save(strcat('models/test_data',cl, num2str(samplesize),'-', num2str(size(centers,2)),'.mat'), 'test_data');
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
