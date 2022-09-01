library(tidyverse)
library(dplyr)
library(ggplot2)
library(textdata)
library(tidytext)
getwd()
t1 = read_csv("~/Downloads/train.csv")
test1 = read_csv("~/Downloads/test.csv")
test1$CRIMETYPE=NA
t1=rbind(t1,test1)
t1[t1$NARRATIVE=='',"NARRATIVE"]<- 'S'
crime_data
corpus = Corpus(VectorSource(t1$NARRATIVE))
corpustest=Corpus(VectorSource(test1$NARRATIVE))
corpus[[1]][1]
t1$CRIMETYPE[1]
corpus = tm_map(corpus, PlainTextDocument)
corpustest = tm_map(corpustest, PlainTextDocument)

corpus = tm_map(corpus, tolower)
corpustest = tm_map(corpustest, tolower)

corpus = tm_map(corpus, removePunctuation)
corpustest = tm_map(corpustest, removePunctuation)

corpus = tm_map(corpus, removeWords, c("unk", stopwords("english")))
corpustest = tm_map(corpustest, removeWords, c("unk", stopwords("english")))

corpus = tm_map(corpus, stemDocument)
corpustest = tm_map(corpustest, stemDocument)
corpus[[1]][1]
frequencies = DocumentTermMatrix(corpus)
frequenciestest = DocumentTermMatrix(corpustest)

sparse = removeSparseTerms(frequencies, 0.995)
sparsetest = removeSparseTerms(frequenciestest, 0.995)

tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(tSparse))

testSparse = as.data.frame(as.matrix(sparsetest))
colnames(testSparse) = make.names(colnames(tSparse))
'testSparse$crimetype=t1$CRIMETYPE'

tSparse$crimetype = t1$CRIMETYPE
prop.table(table(tSparse$crimetype)) 
prop.table(table(testSparse$crimetype)) 
library(caTools)
set.seed(100)
split = sample.split(tSparse$crimetype, SplitRatio = 0.8)
trainSparse = subset(tSparse, split==TRUE)
testSparse = subset(tSparse, split==FALSE)
library(randomForest)
set.seed(100)
trainSparse$crimetype = as.factor(trainSparse$crimetype)
testSparse$crimetype = as.factor(testSparse$crimetype )

RF_model = randomForest(crimetype ~ ., data=trainSparse)
predictRF = predict(RF_model, newdata=testSparse)
table(testSparse$crimetype, predictRF)
table(tSparse$crimetype, predictRF)
tSparse$crimeType


PassengerId= t1$id
output.df=as.data.frame(PassengerId)
submission = data.frame(PassengerId = 100001:200000, CRIMETYPE = predictRF)
write.csv(submission, file = "prediction.csv", row.names = FALSE)

ompletecrimerecords <- na.omit(crime_data) 
db_new=unnest_tokens(tbl=ompletecrimerecords,input=NARRATIVE,output=word)
words
data(stop_words)
stp_wrds=get_stopwords(source='smart')
db_new=anti_join(db_new,stp_wrds,by='word')
bing=get_sentiments(lexicon='bing')
db_bing=inner_join(db_new,bing,by='word')
frequencies = DocumentTermMatrix(db_bing)
db_bing=count(db_bing,CRIMETYPE,sentiment)
#manipulate data
db_bing=spread(key=sentiment,value=n,fill=0,data=db_bing)
db_bing=mutate(sentiment=positive-negative,.data=db_bing)

