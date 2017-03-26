##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/test_img1.jpg "first try"
[image2]: ./output_images/HOG_example.jpg "HOG example"
[image3]: ./output_images/
[image4]: ./output_images/test_img_C1.jpg "Using different C=0.01"
[image5]: ./output_images/test_img_heatmap.jpg "boxed with heat map "
[image6]: ./output_images/test_img_heatmap_onebox.jpg " combine into one box"
[image7]: ./output_images/
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2 code cell of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images. and import hog from skimage.feature. Using hog to create `get_hog_features()` function which is come from the lesson.

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=6`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

After testing with different HOG parameters using images and video, I choose this parameters to get as many window of the car and  as little false positive, also consider for the speed to process the images and video try to keep window under 200 per images. 
Here is the parameters I choose.  
```
color_space = "YCrCb" 
orient =9 
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32,32)
hist_bins =32
spatial_feat = True
hist_feat = True
hog_feat = True
```
####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After choosing the HOG parameters, I use it to train the classifier. The code in the ipython notebook 18 code cell.
before I feed the features into the Support Vector Machines, I first Scale it using StandardScaler() from sklearn. Then split the data into training set and testing set using train_test_split() from sklearn.I start to train my classifier using svc = LinearSVC(C=0.001), and i realize the test accuary only have around 85%.I trained a linear SVM using svc = LinearSVC(C=0.00001) and the test accuary has 99% which is good to use.
I also try different feed into decision tree like random forest, but the speed to train is longer than svm but the accuary does not improve a lot. So at the end, i choose the Support vector machines using C = 0.00001 as my classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the Sliding window and search, I also use the code from the lesson. I choose to 1.3 for scales after testing many times with different number between 1 to 2. I find that the result of 1.3 give me a good output on the video than other.



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


Here is the result when SVM using C=0.01. I trying different number and check the result, as the number get smaller, the false positive is less.
![alt text][image4]

I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Keeping 20frame of heatmap and add together. then using threshold to clean up some false positive. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` :



### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap :
![alt text][image6]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipline stills not perfrom too stable. I try to adjust the scale and increase the frame to add on. It has difficult when the car get to far. And still having some false positive after using more than one frame. 
To improve the pipline, we can try to use different way like YOLO or SSD, to have the better result of locate the car and aviod false positive.

