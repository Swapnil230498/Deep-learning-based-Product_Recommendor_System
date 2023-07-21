Product Recommender System

This self-initiated project entails the creation and deployment of a product recommender system using Streamlit and the ResNet50 Convolutional Neural Network (CNN) model. The system allows users to upload an image to the dedicated website, and based on this input, the system automatically recommends the top 5 products that closely resemble the uploaded image.
Plan of Approach:

    Import Model: For this project, we leverage the power of the ResNet CNN model. ResNet is a well-established pre-trained model developed by computer scientists and data scientists. It has been trained on a vast dataset of various images from IMAGENET, boasting remarkable accuracy. Employing ResNet ensures superior performance, surpassing what could be achieved with a custom CNN.

    Extract Features: The recommender system requires an image of (224, 224, 3) resolution to find matching products for recommendations. However, manually comparing the uploaded image with all 44,000 images used to train ResNet is infeasible. Instead, we extract primitive or complex features from both the 44,000 images and the uploaded image using our ResNet CNN model. By comparing these extracted features, the task becomes manageable. Each image will be represented by a feature vector containing 2048 features. Consequently, we will have a feature matrix of size (44,000 x 2048), with rows corresponding to all the images ResNet was trained on, and a (1 x 2048) feature vector for the uploaded image.

    Generate Recommendations: The recommender system calculates the Euclidean distance between the feature vector of the uploaded image and the feature vectors of all the images in the feature matrix. We utilize k-nearest neighbors (knn) from the scikit-learn library to perform this distance calculation. The rows with the closest distances to the feature vector of the uploaded image are considered our recommendations and are presented to the user.

Dataset Used: Fashion Product Images Dataset (Kaggle)

The dataset employed in this project is the Fashion Product Images Dataset obtained from Kaggle. This dataset serves as an essential foundation for training and evaluating the product recommender system, enhancing its ability to suggest relevant and visually similar products to the users.

By leveraging the capabilities of Streamlit, ResNet, and k-nearest neighbors, this product recommender system promises an engaging and user-friendly experience for discovering top product recommendations based on user-uploaded images.

Your feedback and contributions are most welcome, as we continuously strive to improve and enhance the performance and user experience of this product recommender system.
