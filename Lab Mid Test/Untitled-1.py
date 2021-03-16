import cv2 as cv2

image = cv2.imread('kitkat.png')
image_scene = cv2.imread('kitkat_scene.jpg')

cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_scene = cv2.cvtColor(image_scene,cv2.COLOR_BGR2GRAY)

surf_obj = cv2.xfeatures2d.SURF_create()
key, desc = surf_obj.detectAndCompute(image,None)
keyS, descS = surf_obj.detectAndCompute(image_scene,None)

index_param  = dict(algorithm=0)
search_param = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_param,search_param)
matches = flann.knnMatch(desc,descS,2)

matchesM = []

for i in range(len(matches)):
    matchesM.append([0,0])

for index, (first_best, second_best) in enumerate(matches):
    if first_best.distance < 0.7 * second_best.distance :
        matchesM[index] = [1,0]
res = cv2.drawMatchesKnn(
    image,
    key,
    image_scene,
    keyS,
    matches,
    None,
    matchColor=[0,255,0],
    singlePointColor=[255,0,0],
    matchesMask=matchesM
)

cv2.imshow("res",res)
cv2.waitKey(0)
cv2.destroyAllWindows()




