# Product_Recommender_System
This self-project involves creating and deployment of ResNET50 CNN using streamlit. Once an image is uploaded on the respective website, automatic 5 closest suggestions are provided to the user.

# **Plan Of Attack:**

1) ***IMPORT MODEL*** : we are going to use ResNET CNN model for this project. ResNET model is already trained model, made by many computer scientists and data scientists which is trained on various Images from IMAGENET with very great accuracy and hence we use it. We could have used our own CNN but that would have not gived us the accuracy as high as ResNET.

2) ***EXTRACT FEATURES:*** We are going to have image of (224,224,3) resolution image for which we have to find matches for suggesting similar products. ResNET has trained on 44k images and hence we need to compare the uploaded image with all those 44k images pixels by pixels in order to find the similar products. since this becomes near to impossible task, hence we go for comparing primitive or complex features of all the 44k images with the features obtained for the uploaded image (features will be extracted by our ResNET CNN model for both 44k images and uploaded one). Comparing features makes the task easy as we are in total going to have 2048 features for each of the images. Therefore what we will have is a matrix of size (44k x 2048) where rows are all the 44k images on which ResNET is trained and (1 x 2048) vector for uploaded image.

3) ***GENERATE RECOMMONDATIONS:*** In order to generate recommondations we are going to calculate euclidian distance (knn will be used from sklearn to calculate the distance) for all the rows of the feature matrix with the feature vector we got for uploaded image. Whichever rows have closest distance with the feature vector, they will be considered as our recommondations and will be shown to user.

Dataset used: Fashion Product Images Dataset (kaggle).
