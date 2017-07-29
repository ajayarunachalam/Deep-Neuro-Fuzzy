library("e1071")
library("clValid")
library('graphics')

hd = read.table("/home/sumit/Root Repository/R Projects/humidity-l.txt")
ph = read.table("/home/sumit/Root Repository/R Projects/peek_hour-l.txt")
rf = read.table("/home/sumit/Root Repository/R Projects/rainfall-l.txt")
tp = read.table("/home/sumit/Root Repository/R Projects/temp-l.txt")
wd = read.table("/home/sumit/Root Repository/R Projects/wind-l.txt")

data = cbind(hd, ph, rf, tp, wd)
print (dim(data))

Dist <- dist(data, method="euclidean")

cntr <- c(2,3,4,5,6,7,8,9)
v_dunn <- c()

for (i in cntr){
  
  result <- cmeans(hd,i,1000,verbose=TRUE, dist = 'euclidean', method="cmeans", m=1.1)
  di <- dunn(Dist, result$cluster)
  v_dunn <- append(v_dunn, di)
  
}
#print ('result$cluster : ')
#print (result$cluster)

print (v_dunn)

plot(cntr, v_dunn, xlab='# of center', ylab='Dunn Index for humidity')