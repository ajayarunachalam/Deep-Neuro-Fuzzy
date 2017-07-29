library("e1071")
library("clValid")

data(iris)
# print (iris)
x<-rbind(iris$Sepal.Length, iris$Sepal.Width, iris$Petal.Length)
#print (x)
x<-t(x)

Dist <- dist(x,method="euclidean")
cntr <- c(2,3,4,5,6,7,8,9)
v_dunn <- c()

for (i in cntr){
  
  result <- cmeans(x,i,50,verbose=TRUE,method="cmeans")
  di <- dunn(Dist, result$cluster)
  v_dunn <- append(v_dunn, di)
  
}
#print (Dist)
#print (result$cluster)

print (v_dunn)