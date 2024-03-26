# Undergraduate-Y3S3-PCA-based-Face-Recognition
Author: Ng Zheng Jue, Heng Chia Ying, Tan Hong Guan, Ng Rui Qi

* This is a project developed in undergraduate Year 3 - Semester 3
* This repository consists of code to perform real-time face recognition based on the dataset given in the 'Face_Image' directory.
* This repository consists of
  * 1 Jupyter Notebook file [PCA_based_Face_Recognition.ipynb](https://github.com/xinjue37/Undergraduate-Y3S3-PCA-based-Face-Recognition/blob/main/PCA_based_Face_Recognition.ipynb) that consists of overall experimental design, development of method including training and evaluating the classifier and lastly the real-time face recognition system build.
  * [Face_Image](https://github.com/xinjue37/Undergraduate-Y3S3-PCA-based-Face-Recognition/tree/main/Face_Image) folder to store the face image for recognition
  * [Image](https://github.com/xinjue37/Undergraduate-Y3S3-PCA-based-Face-Recognition/tree/main/Face_Image) folder to store image for display in README.md
  * Library required to run the code
```
!pip install -U scikit-learn
!pip install numpy matplotlib pandas
!pip install xgboost catboost lightgbm
```

## Overall Experiment Design
<img src="Image/Experiment_Design.jpg">
 The overall experiment design starts with data collection. In this phase, a total of
140 face images were gathered, which consisted of a total of 29 distinct individuals.
Among these, 40 images originated from four team members, with each member
contributing 10 images. The remaining 100 images were selected from the open-source
Face Recognition Dataset by the University of Massachusetts Amherst. Among 100
images, it consists of 25 individuals, with each candidate having four images in the
dataset. After that, each of the images is renamed using the format
f’{name}_{num}.jpg’. Examples of filenames for a person “Zhang_Ziyi” will be
“Zhang_Ziyi_1.jpg”, “Zhang_Ziyi_2.jpg”, “Zhang_Ziyi_3.jpg” and
“Zhang_Ziyi_4.jpg”.
Then, for each of the individuals, one of the images is extracted for testing and the
remaining image will move to the training dataset. After train-test splitting, the training
dataset consists of 111 images and the testing dataset consists of 29 images. Afterwards,
all the images from the training dataset are saved in the TRAINING_IMG_DIR
directory while images from the testing dataset are saved in the TESTING_IMG_DIR
directory.
Both datasets will go through data preprocessing. This process involves converting
a colour image into a grayscale image, applying a cascade classifier to detect faces
within the grayscale image, applying an image enhancement technique to improve the
quality of the image and lastly resizing the image into 100x100. In this process, 3
different image enhancement techniques (Histogram Equalization, Gaussian
Smoothing, and Laplacian High Pass Filter) will be used to analyze the effect of image
enhancement technique on the eigenfaces and classifiers performance on the testing
dataset.
Subsequently, for the preprocessed training dataset, it is augmented using
horizontal flipping and rotation with 6 different angles [-15, -10, -5, 5, 10, 15].
Therefore, the training dataset is augmented with a factor of 8 and thus has 888 images13
after data augmentation. The preprocessed training dataset after augmentation is saved
into PROCESSED_IMAGE_DIR to ease the visualization of the images.
After that, the training dataset is loaded from PROCESSED_IMAGE_DIR and
PCA is performed to reduce the dimension of the dataset from (n, m) to (n, k) where n
is the number of face images, m = 10000 since the images have been reshaped to
100x100 and k is the number of features retained. By using Kaiser's rule, only the
eigenvectors that have eigenvalues larger than 1 are retained. After performing PCA,
the top k eigenvectors, mean for each of the features in the training dataset, training
dataset in lower dimension and labels for the training dataset have been saved to
the .npy file so that it can be used to perform classification in testing phase without
running again the PCA.
Next, 10 classifiers were built and trained on a training dataset to recognize 29
person’s faces. The classifiers used are as follow:
1. K-nearest neighbors (KNN) Classifier
2. Logistic Regression
3. Gaussian Naive Bayes
4. C-Support Vector Classification (SVC)
5. Linear Discriminant Analysis (LDA)
6. Random Forest Classifier
7. AdaBoost Classifier
8. XGBoost Classifier
9. CatBoost Classifier
10. LightGBM Classifier
Then, the performances of each of the classifiers are evaluated using the accuracy
metrics. The classifier that has the highest accuracy in the testing dataset is set to be
the best classifier among 10 classifiers. If the accuracy of the best classifier in the
testing dataset is less than 50%, the process will start again from data collection to14
investigate any incorrect label or irrelevant image obtained, else, the face classifier
that has the best performance is selected for real-time face detection.
Lastly, a real-time face recognition system will be built. The process involves
utilizing the VideoCapture in cv2 to capture the image in real-time, preprocessing the
image follow the step in the data preprocessing step, following the projection of the
face to a lower dimension, using the best classifiers obtained to recognize the face,
and lastly labelling the detected face on the image.
