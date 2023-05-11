% Clear the screen and the variables
clc; clear;

% Loads the data
emails = readtable('spam.csv');
emails.Category = double(strcmp(emails.Category, 'spam'));

% Clean the data
emails.Message = tokenizedDocument(emails.Message);
emails.Message = lower(emails.Message);
emails.Message = removeStopWords(emails.Message);
emails.Message = removeShortWords(emails.Message, 3);
emails.Message = erasePunctuation(emails.Message);

% Split data into seperate datasets
cv = cvpartition(emails.Category,'HoldOut',0.3);
idx = cv.test;

% Assign the sets
trainData = emails(~idx,:);
testData = emails(idx,:);

bag = bagOfWords(trainData.Message);

trainCounts = encode(bag, trainData.Message);
testCounts = encode(bag, testData.Message);

% Train the model
model = fitcnb(trainCounts,trainData.Category);

% Test the model
prediction = predict(model, testCounts);

% Calculate the accuracy
accuracy = sum(prediction == testData.Category) / length(testData.Category);
disp(accuracy*100);