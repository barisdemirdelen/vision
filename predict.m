function predict()
cl_rgb = 'rgb';
cl_hsv = 'hsv';
cl_opp = 'opponent';
cl = cl_opp;
samplesize = 5;
vocabsize = 800;
numdescr = 2500; %number of descriptors
load_from_file = true;
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
    image_matrix = []
    aps = [];
    for k=1:4
        image_list = []
        [map, ids] = calculate_map(centers, cl, k, all_images_valid, samplesize);
        aps = [aps; map];
        for id_index=1:size(ids)
            image_path = get_image_path(ids(id_index));
            image_list = [image_list; {image_path}];
        end
        image_matrix = [image_matrix {image_list}];
    end
    disp(image_matrix);
    disp(aps);
    total_map = mean(aps);
    disp(total_map);
    
    write_html(cl, vocabsize,numdescr, total_map, aps, image_matrix);
end
end
end

function write_html(cl, vocabsize,numdescr, total_map, aps, image_matrix)
    fileID = fopen(strcat('results/Result-', cl,'-',num2str(vocabsize),'.html'),'w');
    fprintf(fileID,'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Image list prediction</title><style type="text/css">img {width:200px;}</style></head><body><h2>Baris Demirdelen, Helena Rusello, Edwin Lima</h2><h1>Settings</h1><table><tr><th>SIFT step size</th><td>2 px</td></tr><tr><th>SIFT block sizes</th><td>[4 6 8 10] px</td></tr>\n');
    fprintf(fileID,strcat('<tr><th>SIFT method</th><td>', cl, '</td></tr>\n'));
    fprintf(fileID,strcat('<tr><th>Vocabulary size</th><td>', num2str(vocabsize),' words', '</td></tr>\n'));
    fprintf(fileID,strcat('<tr><th>Vocabulary fraction</th><td> 1.0 </td></tr>\n'));
    fprintf(fileID,'<tr><th>SVM training data</th><td>50 positive, 150 negative per class</td></tr><tr><th>SVM kernel type</th><td>Polinomial</td></tr></table>\n');
    fprintf(fileID,strcat('<h1>Prediction lists (MAP: ', num2str(total_map), ')</h1><table><thead><tr>\n'));
    fprintf(fileID,strcat('<th>Airplanes (AP: ',num2str(aps(3)),')</th><th>Cars (AP: ',num2str(aps(4)),')</th><th>Faces (AP: ',num2str(aps(2)),')</th><th>Motorbikes (AP: ',num2str(aps(1)),')</th></tr></thead><tbody>\n'));
    
    images1 = image_matrix{1};
    images2 = image_matrix{2};
    images3 = image_matrix{3};
    images4 = image_matrix{4};
    for i=1:size(images1)
        fprintf(fileID,'<tr>');
        fprintf(fileID,strcat('<td><img src="',images3{i},'"/></td>'));
        fprintf(fileID,strcat('<td><img src="',images4{i},'"/></td>'));
        fprintf(fileID,strcat('<td><img src="',images2{i},'"/></td>'));
        fprintf(fileID,strcat('<td><img src="',images1{i},'"/></td>'));
        fprintf(fileID,'</tr>\n');
    end
     fprintf(fileID,'</tbody></table></body></html>\n');
     fclose(fileID)
end

function [map,ids] = calculate_map(centers, cl, classifier, test_images, samplesize)
load(strcat('models/svm_model',cl, num2str(classifier),'-',num2str(samplesize),'-', num2str(size(centers,2)), '.mat'), 'model');
test_labels = [];
load(strcat('models/test_data',cl, num2str(samplesize),'-', num2str(size(centers,2)),'.mat'), 'test_data');
for k=1:4
    for i=1:samplesize
        if k == classifier
            test_labels = [test_labels; 1];
        else
            test_labels = [test_labels; 0];
        end
    end
end

[prediction, accuracy, prob_values] = svmpredict(test_labels, test_data, model, '-b 1');
probs = prob_values(:,2);
if classifier == 1
    probs = prob_values(:,1);  % classifier 1 is trained reverse
end
[map, ids] = get_map(probs, test_labels, samplesize);
disp(map);
end

function [map, ids]= get_map(prediction_probs, labels, samplesize)
correct = 0;
total = 0;
map = 0;
to_sort = horzcat(prediction_probs, labels);
ids = 1:size(labels,1);
to_sort = horzcat(to_sort, ids');
to_sort = sortrows(to_sort, -1);

prediction = to_sort(:,1);
labels = to_sort(:,2);
ids = to_sort(:,3);

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

function image_path=get_image_path(id)
image_class = ceil(id/50);
image_id = mod(id,50) + 1;
image_name = strcat('img', sprintf('%03d',image_id), '.jpg');
if image_class == 1
    image_class_name = 'motorbikes_test';
elseif image_class == 2
    image_class_name = 'faces_test';
elseif image_class == 3
    image_class_name = 'airplanes_test';
else
    image_class_name = 'cars_test';
end
image_path = strcat('../CalTech4/imageData/', image_class_name, '/', image_name);
end
