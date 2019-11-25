#Uncomment below comments to install packages at firstime
#install.packages("rattle")
#library("rattle")
#
#
# Define wssplot graph funtion to show different number of cluster
# It can suggest appropriate number of cluster
wssplot <- function(data, nc=15, seed=1234){
               wss <- (nrow(data)-1)*sum(apply(data,2,var))
               for (i in 2:nc){
                    set.seed(seed)
                    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
                plot(1:nc, wss, type="b", xlab="Number of Clusters",
                     ylab="Within groups sum of squares")}

# a dataset containing 13 chemical measurements on 178 Italian wine samples is analyzed.
# The data originally come from the UCI Machine Learning Repository

# Load wine data either from rattle package or from UCI URL
#install.packages("NbClust")
#library("NbClust")
#data(wine, package="rattle")

wine <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data")

names(wine) <- c("Type", "Alcohol","MalicAcid","Ash","Alcalinity","Magnesium","Totalphenols","Flavanoids","Nonflavanoid","Proanthocyanins","Color","Hue","dilution","Proline")   
head(wine)
#omit NA
df<-na.omit(df)

# Scale wine data exclude type
df <- scale(wine[-1])  

# Display graph how evaluation of different numbers of cluster
wssplot(df)

# Find the best number of clusters
set.seed(1234)
nc <- NbClust(df, min.nc=2, max.nc=15, method="kmeans")
table(nc$Best.n[1,])

barplot(table(nc$Best.n[1,]),
          xlab="Numer of Clusters", ylab="Number of Criteria",
          main="Number of Clusters Chosen by 26 Criteria")

#Input the best number of k-mean clusters
no_clusters <- readline(prompt="Enter the best number of cluster from the graph")
set.seed(1234)
fit.km <- kmeans(df, as.integer(no_clusters), nstart=25)                           
fit.km$size

fit.km$centers
# procue the table with mean of k-mean 
aggregate(wine[-1], by=list(cluster=fit.km$cluster), mean)

#A cross-tabulation of Type (wine varietal) and cluster membership is given by
ct.km <- table(wine$Type, fit.km$cluster)
ct.km
#install.packages("flexclust")
#library("flexclust")
#Evaluation (0-1) for satisfastion between type and cluster using an adjusted Rank index provided 
randIndex(ct.km)