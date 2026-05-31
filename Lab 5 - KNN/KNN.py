import math
X_train = [[2, 4],[4, 6],[6, 8], [8, 10],[10, 12]]
y_train=[0,0,1,1,1]
def euclidean_distance(p1,p2):
    distance=0
    for i in range(len(p1)):
        distance+=(p1[i]-p2[i])**2
    return math.sqrt(distance)
def knn_predict(X_train,y_train,test_point,k):
    distances=[]
    for i in range(len(X_train)):
        dist=euclidean_distance(X_train[i],test_point)
        distances.append((dist,y_train[i]))
    distances.sort()
    neighbors=distances[:k]
    count0=0
    count1=0
    for _, label in neighbors:
        if label==0:
            count0 += 1
        else:
            count1 += 1
    if count1>count0:
        return 1
    else:
        return 0
test_point=[5,7]
k=3

prediction=knn_predict(X_train, y_train,test_point,k)
print("Test Point:", test_point)
print("Predicted Class:", prediction)