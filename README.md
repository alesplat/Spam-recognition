# Spam-recognition
This project focuses on recognising spam email. Dataset contains most of email as "ham" (normal email) and some "spam" email.

1. Imported the training set and removed some lines that had ; as the only character
2. Converted the text to lowercase, removed the punctuation, removed all english stopwords, stemmed the words
3. Built a document term matrix, so to convert words in numbers
4. Removed spars terms,  maintend only the terms that appear in at least the 3% of the emails (in order to reduce the words that need to be analyzed to build the model)
5. Built the logistic regression model, choosing the predictors through the stepwise regression (command stepAIC)
6. Imported the test set and made the fix that had been done previously for the training set
7. Made prediction on the training data, obtaining a training error rate of around 9%.
8. Performed the AUC in the training set
9. Predicted the results on the test data, with an accuracy of 90%



## Files in the repository
- spam_train.csv, spam_test.csv => train and test set
- spam_test_with_results.csv => test set with a column added containing the prediction obtained ("spam" or "ham")
- Spam classification.R => the code of the project in R language
