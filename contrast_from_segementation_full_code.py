#!/usr/bin/env python
# coding: utf-8

#code to convert image to single channel only form


import cv2
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


#read image
image =  cv2.imread('NJ_WS_03_03_100X_cropped.jpg', cv2.IMREAD_UNCHANGED)
print(image.shape)


img = plt.imshow(image)
plt.show()



#the color change above is because cv2 reads and keep the image as BGR
#whereas matplotlib uses RGB. So for visualisation using matplotlib, it
#is better to convert.
#We are only using the green channel and hence we will not be encoun
#tering any issues eitherway.


#extract green channel
#for mos2 extract red channel as given by Wang et al
green_channel_image = image[:,:,1]
red_channel_image = image[:,:,2]
blue_channel_image = image[:,:,0]
#for blue channel, replace the 1 with 0, and use 2 for red


#create empty image with same shape as image
green_image = np.zeros(image.shape)
blue_image = np.zeros(image.shape)
red_image = np.zeros(image.shape)



#assign green channel of the original image to empty image
green_image[:, :, 1]= green_channel_image
red_image[:, :, 2]= red_channel_image
blue_image[:,:,0] = blue_channel_image


cv2.imwrite('NJ_WS_03_03_100X_cropped_red_channel.png' , red_image)
cv2.imwrite('NJ_WS_03_03_100X_cropped_green_channel.png' , green_image)
cv2.imwrite('NJ_WS_03_03_100X_cropped_blue_channel.png', blue_image)
#print(green_image.shape)
print(red_image.shape)



#just to see how cv2 and plt images look different
#save_test_image = cv2.cvtColor(green_channel_image, cv2.COLOR_BGR2RGB)
#plt.imshow(save_test_image)



#We add the bilteral filter before going to the clustering part 
#to get rid of the noise as mentioned by Sci_Rep_paper



img_file = "NJ_WS_03_03_100X_cropped_red_channel.png"


img = cv2.imread(img_file)



## Bilateral filtering
for ii in range(1):
    img = cv2.bilateralFilter(img,2,1,1)
print(img.shape)



cv2.imwrite('NJ_WS_03_03_100X_cropped_red_channel_bfapplied.png' , img)
print(img.shape)


# In[17]:


#starting the part where we make clusters of various groups
#and see which has a lesser inertia using the elbow plot.
#This time kneedle algorithm can be tried to automatically
#obtain ideal number of clusters.


image = cv2.imread('NJ_WS_03_03_100X_cropped_red_channel_bfapplied.png')
image2 = cv2.imread('NJ_WS_03_03_100X_cropped_red_channel_bfapplied.png')


image = image.reshape((image.shape[1]*image.shape[0],3))
#print(image.shape(R_G_13_04_100x))



#creating an empty list and adding to it the inertia for number 
#of clusters running from 1 to 11



inertia_list=[]
for i in range(1,12):
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(image)
  inertia=kmeans.inertia_
  inertia_list.append(inertia)
print(inertia_list)



#plotting inertia as a function of the number of clusters
fig = plt.figure(figsize=[5,7])
ax = plt.subplot(111)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.plot(list(np.arange(1,12)),inertia_list)
ax.set_xlabel('Number of Clusters, N', fontsize=16)
ax.set_ylabel('Inertia', fontsize=16)
ax.set_title('Elbow Search for Fig.1.c', fontsize=17)
ax.grid('on')
ttl = ax.title
ttl.set_weight('bold')
plt.savefig('NJ_WS_03_03_100X_cropped_red_channel_bfapplied_elbow.png',dpi=600)
plt.show()



#using the kneedle algorithm to autodetect the elbow point
'''

num_of_clusters = np.arange(1,12,1)


print(num_of_clusters)


from kneed import KneeLocator


kneedle = KneeLocator(num_of_clusters, inertia_list, S=1.0, curve="convex", direction="decreasing",interp_method='polynomial', online=True)



#The optimal number of clusters come from the elbow point
k = round(kneedle.elbow, 3)
print("the elbow point is ",k)


print(round(kneedle.knee, 3))


kneedle.plot_knee()
plt.savefig('R_MoS2_02_05_100X_6p5_test_titleremoved.png',dpi=600)
plt.show()


#image2 = image.reshape((-1,3))


#image2D = image.reshape(image.shape[0]*image.shape[1],image.shape[2])


#image2 = np.float32(image2)
'''


k = int(input("enter the number of clusters needed "))

kmeans = KMeans(n_clusters=k)
kmeans.fit(image)


clustered = kmeans.cluster_centers_[kmeans.labels_]


segmented_image = clustered.reshape(image2.shape[0],image2.shape[1],image2.shape[2])


#plt.imshow(segmented_image)


cv2.imwrite('NJ_WS_03_03_100X_cropped_red_channel_bfapplied_segmented.png' , segmented_image)

#get_ipython().run_line_magic('matplotlib', 'tk')


img = cv2.imread("./NJ_WS_03_03_100X_cropped_red_channel_bfapplied_segmented.png")


#click region of substrate to get substrate pixel value


def click_event(event, x, y, flags, params):
    '''
    Left Click to get the x, y coordinates.
    Right Click to get BGR color scheme at that position.
    '''
    text = ''
    font = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        text = str(x) + "," + str(y)
        color = (0, 255, 0)
    elif event == cv2.EVENT_RBUTTONDOWN:
        b = img[y, x, 0]
        click_event.g = img[y, x, 1]
        click_event.r = img[y, x, 2]
        print("this is the green pixel value", click_event.g)
        print("this is the red pixel value", click_event.r)
        text = str(b) + ',' + str(click_event.g) + ',' + str(click_event.r)
        color = (0, 0, 255)
    cv2.putText(img, text, (x, y), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.imshow('image', img)



print(img)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey()
cv2.destroyAllWindows()



r_sub = click_event.r
g_sub = click_event.g
print('the red pixel value of the substrate : ', r_sub)
print('the green pixel value of the substrate : ', g_sub)


#click region of sample to get sample pixel value


def click_event(event, x, y, flags, params):
    '''
    Left Click to get the x, y coordinates.
    Right Click to get BGR color scheme at that position.
    '''
    text = ''
    font = cv2.FONT_HERSHEY_COMPLEX
    color = (255, 0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ",", y)
        text = str(x) + "," + str(y)
        color = (0, 255, 0)
    elif event == cv2.EVENT_RBUTTONDOWN:
        b = img[y, x, 0]
        click_event.g = img[y, x, 1]
        click_event.r = img[y, x, 2]
        print("this is the green pixel value", click_event.g)
        print("this is the red pixel value", click_event.r)
        text = str(b) + ',' + str(click_event.g) + ',' + str(click_event.r)
        color = (0, 0, 255)
    cv2.putText(img, text, (x, y), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.imshow('image', img)



print(img)
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey()
cv2.destroyAllWindows()


#g_sam = click_event.g
#print(g_sam)
r_sam = click_event.r
g_sam = click_event.g
print('the red pixel value of the selected sample region is ', r_sam)
print('the green pixel value of the selected sample region is ', g_sam)


contrast_r = (r_sub - r_sam)/r_sub
contrast_g = (g_sub - g_sam)/g_sub
print("The contrast in red channel is ", contrast_r)
print("The contrast in green channel is ", contrast_g)



