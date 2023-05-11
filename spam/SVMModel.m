% Clear the screen and the variables
clc; clear;

% Loads the data
emails = readtable('spam.csv');
content = emails.Message;
label = emails.Category;
label = strcmp(label, 'spam');

% Clean the data
features = tokenizedDocument(content);
features = lower(features);
features = removeStopWords(features);
features = removeShortWords(features, 3);
features = erasePunctuation(features);
bag = bagOfWords(features);
bag = removeInfrequentWords(bag, 4);
email = bag.Counts;
words = full(email);

% Split data into seperate datasets
cv = cvpartition(size(words,1),'HoldOut',0.3);
idx = cv.test;

% Assign the sets
trainData = words(~idx,:);
testData = words(idx,:);
trainLabel = label(~idx);
testLabel = label(idx);

% Train the model
model = fitcsvm(trainData, trainLabel);

% Test the model
[label, score] = predict(model,testData);

% Display accuracy
accuracy = sum(label == testLabel) / length(testLabel);
disp(accuracy*100);