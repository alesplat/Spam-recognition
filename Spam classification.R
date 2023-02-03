# load text mining package
library(tm)
library(dplyr)
library(leaps)
library(dispRity)
library(Rcmdr)
library(ROCR)
library(MASS)

spam_train <- read.csv2("~/Buy a Beer competition/spam_train.csv")
spam_train$class <- as.factor(spam_train$class)
spam_train[spam_train$class == "", ] #cosi mi trovo gli elementi strani, da rimuovere
spam_train <- spam_train[c(-1051, -4158), ] #rimuovo questi elementi
spam_train$class <- droplevels(spam_train$class)
summary(spam_train)
str(spam_train)
table(spam_train$class)

# Build a new corpus variable called corpus
corpus <- VCorpus(VectorSource(spam_train$email))
# convert the text to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, PlainTextDocument)
# remove all punctuation from the corpus
corpus <- tm_map(corpus, removePunctuation)
# remove all English stopwords from the corpus
corpus <- tm_map(corpus, removeWords, stopwords("en"))
# stem the words in the corpus
corpus <- tm_map(corpus, stemDocument)

# Build a document term matrix from the corpus
dtm = DocumentTermMatrix(corpus)
dtm

# Remove sparse terms (that don't appear very often), in order to consider just the terms that appear in at least 5% of the documents
spdtm = removeSparseTerms(dtm, 0.97) #0.97 is the maximal allowed sparsity
spdtm
#In the sense of the sparse argument to removeSparseTerms(), sparsity refers to the threshold of relative document frequency for a term, above which the term will be removed. Relative document frequency here means a proportion. As the help page for the command states (although not very clearly), sparsity is smaller as it approaches 1.0. (Note that sparsity cannot take values of 0 or 1.0, only values in between.)
#For example, if you set sparse = 0.99 as the argument to removeSparseTerms(), then this will remove only terms that are more sparse than 0.99. The exact interpretation for sparse = 0.99 is that for term $j$, you will retain all terms for which $df_j > N * (1 - 0.99)$, where $N$ is the number of documents -- in this case probably all terms will be retained (see example below).
#Near the other extreme, if sparse = 0.01, then only terms that appear in (nearly) every document will be retained


# Convert spdtm to a data frame
emailsSparse <- as.data.frame(as.matrix(spdtm))
# make variable names of emailsSparse valid i.e. R-friendly (to convert variables names starting with numbers)
colnames(emailsSparse) <- make.names(colnames(emailsSparse))
sort(colSums(emailsSparse), decreasing = TRUE)

# Add dependent variable to this dataset
emailsSparse$class <- as.integer(spam_train$class)
str(emailsSparse)
summary(emailsSparse)
# most frequent words in ham:
sort(colSums(subset(emailsSparse, class == 1))) #class = 3860 perche le parole non spam (in ham) sono 3860
# most frequent words in spam:
sort(colSums(filter(emailsSparse, class == 2))) #class = 1194 perche le parole spam sono 597 x 2 che è il numero con cui as.integer ha convertito il valore "spam"



# convert the dependent variable to a factor
emailsSparse$class = as.factor(emailsSparse$class)
summary(emailsSparse)


# Build a logistic regression model (individuando le variabili giuste con la stepwise, ci mette qualche secondo)
spamLog = glm(emailsSparse$class ~., data=emailsSparse, family="binomial") %>% stepAIC(trace = FALSE)




#Sistemazione del test set (riprende in gran parte cio che ho fatto gia nel training)
spam_test <- read.csv2("~/Buy a Beer competition/spam_test.csv")
spam_test$id_number <- NULL
# Build a new corpus variable called corpus
corpus <- VCorpus(VectorSource(spam_test$email))
# convert the text to lowercase
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, PlainTextDocument)
# remove all punctuation from the corpus
corpus <- tm_map(corpus, removePunctuation)
# remove all English stopwords from the corpus
corpus <- tm_map(corpus, removeWords, stopwords("en"))
# stem the words in the corpus
corpus <- tm_map(corpus, stemDocument)


# Build a document term matrix from the corpus
dtm2 = DocumentTermMatrix(corpus)
#dtm2

# Remove sparse terms (that don't appear very often), in order to consider just the terms that appear in at least 5% of the documents
#spdtm2 = removeSparseTerms(dtm, 0.97) #0.97 is the maximal allowed sparsity
#spdtm2
#In the sense of the sparse argument to removeSparseTerms(), sparsity refers to the threshold of relative document frequency for a term, above which the term will be removed. Relative document frequency here means a proportion. As the help page for the command states (although not very clearly), sparsity is smaller as it approaches 1.0. (Note that sparsity cannot take values of 0 or 1.0, only values in between.)
#For example, if you set sparse = 0.99 as the argument to removeSparseTerms(), then this will remove only terms that are more sparse than 0.99. The exact interpretation for sparse = 0.99 is that for term $j$, you will retain all terms for which $df_j > N * (1 - 0.99)$, where $N$ is the number of documents -- in this case probably all terms will be retained (see example below).
#Near the other extreme, if sparse = 0.01, then only terms that appear in (nearly) every document will be retained


# Convert spdtm to a data frame
spam_test_datamatrix <- as.data.frame(as.matrix(dtm2))
# make variable names of emailsSparse valid i.e. R-friendly (to convert variables names starting with numbers)
colnames(spam_test_datamatrix) <- make.names(colnames(spam_test_datamatrix))
sort(colSums(spam_test_datamatrix), decreasing = TRUE)



#prediction on training data
glm.probs <- predict(spamLog, type= "response")
pred.glm <- rep(0, length(glm.probs))
pred.glm[glm.probs > 0.5] <- 1
table(pred.glm, spam_train$class)
(368+74)/nrow(spam_train) #confusion matrix


#AUC training set logistic regression
predictionTrainLog = prediction(glm.probs, spam_train$class)
as.numeric(performance(predictionTrainLog, "auc")@y.values)



#Prediction on testing data
test.probs = predict(spamLog, newdata=spam_test_datamatrix, type="response") #spamLog e' il glm fatto sul training, spam_test_datamatrix e' il test set
pred.TEST <- rep(0, length(test.probs))
pred.TEST[test.probs > 0.5] <- "spam"
pred.TEST[test.probs < 0.5] <- "ham"
table(pred.TEST)



#write in the csv test data
tmp <- data.frame(spam_test$email, pred.TEST)
write.table(tmp, file = "~/Buy a Beer competition/spam_test_with_results.csv", row.names = F)




