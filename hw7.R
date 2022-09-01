library(tidyverse)
library(dplyr)
library(ggplot2)
library(textdata)
library(tidytext)
getwd()
crimedata = read_csv("/Users/bilvikakasturi/PycharmProjects/Datascience/train.csv")
#stopwords help you to remove the common words/unnecessary words who are in both btfv or burg
#changing categorical to numerical.

crimedata$NARRATIVE=as.character(crimedata$NARRATIVE)
BEDROOM_per_CRIME = crime_words_no_stop %>% 
  group_by(CRIMETYPE)  %>% filter(word=="veh")   %>%  dplyr::count(word,sort=T)
BEDROOM_per_CRIME
BEDROOM_per_bedroom = crime_words_no_stop %>% 
  group_by(CRIMETYPE)  %>% filter(word=="bedroom")   %>%  dplyr::count(word,sort=T)
BEDROOM_per_bedroom
words = crimedata %>%
  unnest_tokens(word,NARRATIVE )
words
#use tf idf for stopwords prediction(Google it)
#reduce dimension and keep informative words by removing stop words.
data(stop_words)

crime_words_no_stop <- words %>%
  anti_join(stop_words)


crime_words_no_stop %>% group_by(word) %>% summarise(count=n()) %>% arrange(desc(count)) %>%print(n = 100)

crime_words_no_stop %>% group_by(word) %>% summarise(count=n()) %>% arrange((count))
#TO GET THE summary of your dataframe/tibble. 
summary(crime_words_no_stop)
crime_words_no_stop
crimedata$Veh=grepl("VEH",crimedata$NARRATIVE)
crimedata$bedroom=grepl("bedroom",crimedata$NARRATIVE)
crimedata$car=grepl("car",crimedata$NARRATIVE)
crimedata$driver=grepl("driver",crimedata$NARRATIVE)
crimedata$passenger=grepl("passenger",crimedata$NARRATIVE)
crimedata$type=grepl("BTFV",crimedata$CRIMETYPE)
crimedata <- na.omit(crimedata) 
xy <- data.frame(x=crimedata$X, y=crimedata$Y)

model_glm=glm(type~Veh+bedroom+driver+passenger,data=crimedata)
glm_probs = data.frame(probs = predict(model_glm, type="response"))
head(glm_probs)
crimedatatest = read_csv("/Users/bilvikakasturi/PycharmProjects/Datascience/test.csv")
glm.prediction = predict(model_glm, newdata=crimedatatest, type='response')
summary(model_glm)

predict(model_glm,crimedata)
# Transformed data
pj <- project(xy, proj4string, inverse=TRUE)
latlon <- data.frame(lat=pj$y, lon=pj$x)
print(latlon)
crimedata$lat=pj$y
crimedata$long=pj$x
'mapview(crimedata, xcol = "long", ycol = "lat", crs = 4269, grid = FALSE)
sbux_sf <- st_as_sf(crimedata, coords = c("long", "lat"),  crs = 4326)
mapview(sbux_sf, map.types = "Stamen.Toner") '
crimedata %>%
  group_by(id) %>%
  mutate(count=n()) %>%
  filter(count>1)
(t<-crimedata %>% 
    group_by(CRIMETYPE) %>%
    summarize(count=n()) %>%
    arrange(desc(count)))
library(scales)
crimedata %>% 
  group_by(CRIMETYPE) %>%
  summarise(count=n()) %>%
  ggplot(aes(x = reorder(CRIMETYPE,count), y = count)) +
  geom_bar(stat = "identity", fill = "#756bb1") +
  labs(x ="Crimes", y = "Number of crimes", title = "Crimes in LA") + 
  scale_y_continuous(label = comma) +
  coord_flip()

library(randomForest)
rf.model = randomForest(type ~ Veh+bedroom+BEDROOM_per_bedroom+BEDROOM_per_CRIME+passenger, data = crimedata, type = 'response')
print(rf.model$confusion)
rf.prediction = predict(rf.model, crimedatatest)

prediction = predict(rf.model, newdata = Titanic.test)
PassengerId= Titanic.test$PassengerId
output.df=as.data.frame(PassengerId)
submission = data.frame(PassengerId = 892:1309, Survived = prediction)
write.csv(submission, file = "prediction.csv", row.names = FALSE)