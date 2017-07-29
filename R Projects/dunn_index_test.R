library("clValid")

data(mouse)
express <- mouse[1:25,c("M1","M2","M3","NC1","NC2","NC3")]

#print (mouse)
#print (express)
rownames(express) <- mouse$ID[1:25]
#print (rownames(express))
Dist <- dist(express,method="euclidean")
#print (Dist)
clusterObj <- hclust(Dist, method="average")
print (clusterObj)
nc <- 2 ## number of clusters      
cluster <- cutree(clusterObj,nc)
print ('Cluster : ')
print (cluster)
#print (cluster)
dunn(Dist, cluster)
