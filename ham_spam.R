require(caret)
require(tm)
require(wordcloud)
require(e1071)
require(MLmetrics)

sms_raw <-  read.csv(file.choose(), stringsAsFactors = FALSE)
str(sms_raw)


# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

# examine the type variable more carefully
str(sms_raw$type)


table(sms_raw$type)

library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
# examine the sms corpus
print(sms_corpus)

inspect(sms_corpus[1:2])


as.character(sms_corpus[[1]])

lapply(sms_corpus[1:2], as.character)

# clean up the corpus using tm_map()
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# show the difference between sms_corpus and corpus_clean
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])


sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation


# illustration of word stemming
library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

# examine the final clean corpus
lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

# create a document-term sparse matrix
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)


# creating training and test datasets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

# also save the labels
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

# check that the proportion of spam is similar
prop.table(table(sms_train_labels))

prop.table(table(sms_test_labels))

library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

spam <- subset(sms_raw, type == "spam")
ham  <- subset(sms_raw, type == "ham")
wordcloud(ham$text, max.words = 50, scale = c(3, 0.5))
wordcloud(spam$text, max.words = 50, scale = c(3, 1))

# save frequently-appearing terms to a character vector
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]



convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)


library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)
sms_train_pred <- predict(sms_classifier, sms_train)
head(sms_train_pred)

sms_test_pred <- predict(sms_classifier, sms_test)
head(sms_test_pred)

library(gmodels)
# CrossTable(sms_train_pred, sms_train_labels,
#            prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
#            dnn = c('predicted', 'actual'))
# detach()
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))


sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
sms_classifier3 <- naiveBayes(sms_train, sms_train_labels, laplace = 3)
sms_test_pred3 <- predict(sms_classifier3, sms_test)
CrossTable(sms_test_pred3, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
# So on increasing the laplace value the prediction of Ham is increasing and that of Spam is decreasing, 
# And thus it seems to overfit the model.










































































































