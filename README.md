# Drug-Activity-Prediction

**Overview and Assignment Goals:**

- Use/implement a feature selection/reduction technique. 
- Experiment with various classification models.
- Think about dealing with imbalanced data.
- Implement an ensemble classifier with boosting.
- Use F1 Scoring Metric

**Detailed Description:**
PART 1 - Develop predictive models that can determine given a particular compound whether it is active (1) or not (0). Drugs are typically small organic molecules that achieve their desired activity by binding to a target site on a receptor. The first step in the discovery of a new drug is usually to identify and isolate the receptor to which it should bind, followed by testing many small molecules for their ability to bind to the target site. This leaves researchers with the task of determining what separates the active (binding) compounds from the inactive (non-binding) ones. Such a determination can then be used in the design of new compounds that not only bind, but also have all the other properties required for a drug (solubility, oral absorption, lack of side effects, appropriate duration of action, toxicity, etc.). The goal of this competition is to allow you to develop predictive models that can determine given a particular compound whether it is active (1) or not (0). As such, the goal would be develop the best binary classification model. A molecule can be represented by several thousands of binary features which represent their topological shapes and other characteristics important for binding. Since the dataset is imbalanced the scoring function will be the F1-score instead of Accuracy. 

PART 2 - Also, we will use the same dataset, but implement (i.e. write your own code for) an ensemble classifier using Boosting, e.g. AdaBoost. You are free to use libraries or packages for data pre-processing as in HW2, as well as your choice of base classifiers. However, you must write your own code for the boosting and the ensemble part.

**Caveats:**
- Remember not all features will be good for predicting activity. Think of feature selection, engineering, reduction (anything that works)
- The dataset has an imbalanced distribution i.e., within the training set there are only 78 actives (+1) and 722 inactives (0). No information is provided for the test set regarding the distribution.
- Use your data mining knowledge till now, wisely to optimize your results.

**Data Description:**
The training dataset consists of 800 records and the test dataset consists of 350 records. We provide you with the training class labels and the test labels are held out. The attributes are binary type and as such are presented in a sparse matrix format within train.dat and test.dat

- Train data: Training set (a sparse binary matrix, patterns in lines, features in columns: the number of the non-zero features are provided with class label 1 or 0 in the first column.
- Test data: Testing set (a sparse binary matrix, patterns in lines, features in columns: the number of non-zero features are provided).
- Format example: A sample submission with 350 entries randomly chosen to be 0 or 1.

**APPROACH FOLLOWED:**
1.	Loaded the training dataset and the test data set into data frames using the file opening and redlines techniques in python. (Provided local file paths)
2.	Generate the sparse matrices out of the test data and training data using the input files.
3.	Now we have to store the classes of training set into a separate list and also the training labels set.
4.	Now, we have to also reduce the dimensionality of the sparse matrix generated out of the training and test datasets so that the predictions can be more accurate.
5.	Techniques like SVD which is Singular Value Decomposition and PCA i.e., Principal Component Analysis has been choosed to reduce dimensionality.
6.	SVD is used mostly for sparse data and reduces the dimensions by decreasing the input variables of dataset. 
7.	Now, we will have to choose the technique that gives more F1 score that is SVD has been choosed. SVD gave the score 0.78 where PCA gave the score as 0.72.
8.	For SVD, different components gave different F1 score values. The number of n components and random state (=42) has been choosen as it gave more score that is 0.78. 
9.	SVD Comparisons has been listed below:

<img width="383" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/d6b4be81-6d78-4f20-a001-df3ff50b482c">

10. Also, we need to solve the data bias i.e., solve imbalance data problem using techniques like SMOTE, SMOTEENN, SMOTE Tomek and Random Over Sampler from imblearn packages in python. 
11.	By using these above techniques the F1 score has been reduced to 0.67 and without using these techniques the F1 score is 0.78.
12.	We can use sklearn library to use the python packages for classification techniques and then use predict() method to calculate the F1 score
13.	We need to perform classification techniques like Decision tress, Naïve bayes and Neural Networks to calculate the F1 score.
14.	Using the above classifiers different F1 scores has been calculated like below:

<img width="394" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/77bef5f6-e875-40d1-903a-bf8ed4624cbe">

15.	Based on the above table and calculations, Naive Bayes - Bernoulli's Classification technique has been choosed as a classifier because it gave the highest F1 score which is 0.78.
16.	Finally the predictions with 350 rows has been written into the output file.

-- Decision Trees, Naïve Bayes, and Neural Networks are popular classification techniques used in machine learning. Each technique has its own strengths and weaknesses, and their suitability depends on the nature of the data and the problem at hand.

**Decision Trees:**
Decision trees are simple yet powerful classification models that use a tree-like structure to make decisions. The tree is built by recursively splitting the data based on different features, creating branches that represent different decision paths. The splits are determined by certain criteria, such as entropy or Gini impurity, to maximize information gain and minimize impurity. Decision trees are easy to interpret and can handle both numerical and categorical data. However, they can be prone to overfitting if the tree becomes too complex.

**Naïve Bayes:**
Naïve Bayes is a probabilistic classification technique based on Bayes' theorem with the assumption of independence between features. It calculates the probability of a sample belonging to a particular class given its feature values. Despite the naïve assumption, Naïve Bayes often performs well in practice, especially when the independence assumption holds approximately or the feature space is high-dimensional. It is computationally efficient and can handle large datasets. However, it may not capture complex relationships between features and is sensitive to feature correlations.

**Neural Networks:**
Neural Networks are a versatile and powerful class of models inspired by the structure and functioning of the human brain. They consist of interconnected nodes (neurons) organized in layers, including an input layer, one or more hidden layers, and an output layer. Each neuron applies a nonlinear transformation to its inputs and passes the result to the next layer. Neural Networks can learn complex patterns and relationships in the data, making them suitable for a wide range of classification tasks. However, they require a large amount of labeled data and can be computationally expensive to train. They also have a "black box" nature, making them less interpretable compared to other techniques.

It's important to note that these are just three examples of classification techniques, and there are many other algorithms and models available. The choice of technique depends on various factors, including the characteristics of the data, the complexity of the problem, computational resources, and interpretability requirements.

**PART 2 - BOOSTING ALGORITHMS**

**DIRECTION TO UPLOAD IN JUPITER NOTEBOOK**

- Upload the .ipynb file in Jupyter notebook then open the code
- Train data path used: /Users/sanjanagovindu/Downloads/584_HW3/DM Asst3/traindata.csv
- Test data path used: /Users/sanjanagovindu/Downloads/584_HW3/DM Asst3/testdata.csv
- We need to change the paths of test and train file and then run the code 
- Then we need to run each and every part and calculate the predictions file 

**IMPLEMENTATION**

HW3-G01380306.py - This file contains the main classifier code along with feature selection and dimensionality reduction like pca and svm. It also balances the data using SMOTE and SMOTETek. After that, the Adaboost class is implemented where it is used to boost the perfomance with different M values. We can run the code by using the command - py HW3-G01380306.py (in terminal at file path)

**OBJECTIVES**
Objectives of the assignment:
• Implementation of feature selection or dimensionality reduction technique
• Dealing with Imbalanced data in the datasets
• Implementation of boosting enabler like Adaboost
• Calculation of F1 score using the data
• Perform cross validation

**DESCRIPTION OF THE DATA AND PROBLEM**
Problem statement: The assignment is about Drug Activity prediction where we have to calculate the F1 score (as the given dataset is imbalanced) using the predictions we made using the given training and test dataset files. The given training dataset has an imbalanced distribution 78 actives (1) and 722 inactive (0). The goal is to allow to develop predictive models that can determine given a particular compound whether it is active (1) or not (0) by implementing an enabler classifier using Boosting techniques like Adaboost.

**OBSERVATIONS**
From the experiment in the code, we can say that the dimensionality reduction techniques have good impact on the performance of the code. SVM technique gave better results when compared to PCA. SMOTETek gave good results compared to SMOTE. The value of M as 200 is giving better results in the experiment.

**APPROACH FOLLOWED:**
1. Loaded the training dataset and the test data set into data frames using the file opening and redlines techniques in python. (Provided local file paths)
2. Generate the sparse matrices out of the test data and training data using the input files.
3. Now we must store the classes of training set into a separate list and also the training labels set.
4. Now, we must also reduce the dimensionality of the sparse matrix generated out of the training and test datasets so that the predictions can be more accurate.
5. Techniques like SVD which is Singular Value Decomposition and PCA i.e., Principal Component Analysis has been choosed to reduce dimensionality.
6. SVD is used mostly for sparse data and reduces the dimensions by decreasing the input variables of dataset. SVD is used to reduce the reduce the complexity of the code.
7. Now, we will have to choose the technique that gives more F1 score that is SVD has been choosed. SVD gave the score 0.69 where PCA gave the score as 0.64.
8. For SVD, different components gave different F1 score values. The number of n components and random state (=42) has been choosen as it gave more score that is 0.69.
9. SVD Comparisons has been listed below:

<img width="543" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/01bba7a4-9a60-4f36-94ce-2b859015b724">

10. Also, we need to solve the data bias i.e., solve imbalance data problem using techniques like SMOTE and SMOTE Tomek from imblearn packages in python. SMOTETomek is used for upsampling and downsampling. SMOTE is an oversampling method where as Tomek is an undersampling method. So SMOTETomek combines two of them and deals with data imbalance. For better performance it has been selected.
11. We can use sklearn library to use the python packages for classification techniques and then use predict () method to calculate the F1 score
12. We need to perform classification techniques using boosting techniques like adaboost to get the F1 Score

**ADABOOST CLASSIFIER:**
Any machine learning algorithm's performance may be improved with AdaBoost. When training weak learners, it works best. These are models whose categorization accuracy is somewhat better than random chance. According to the formula below, this meta-classifier will assign a prediction for each observation based on a weighted majority vote. The adaboost algorithm will fit sequence of stumps through boosting series of iterations by M value. In each iteration, we will give more weight to those iterations that has been misclassified in previous iterations. Then we can calculate the error rate and alpha value from minimizing the exponential function loss. The.predict() function accepts an array-like matrix of explanatory variables as an input, much like the.fit() method does. This calculates the prediction for each observation and stored in a dataframe. The decision tree model has been used as its is easy to understand and it’s a weak model and in each oteration we get different decision tree. Then the final predictions has been stored in the output file.

<img width="311" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/72b5e5d8-b650-4b26-bfc7-c28ef1f614d9">

<img width="558" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/c519255a-2911-4b3d-aa56-8f781145395f">

13. Using the above classifier different F1 scores has been calculated like below for different M values:

<img width="528" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/52b4d915-cab9-4f53-8ca6-f6c83f137bef">

14. Finally, the predictions with 350 rows have been written into the output file and submitted on Miner 2 platform to calculate the score.
15. After that finally the cross validation has been implemented.

<img width="615" alt="image" src="https://github.com/sanjana-govindu/Drug-Activity-Prediction/assets/54507596/78a0aa01-5d72-4a65-929a-102b7e7ca6fb">

**LIBRARIES USED:**
- Python libraries like NumPy and Pandas have been used in this homework.
- NumPy supports multi-dimensional arrays that perform matrix and vector operations, and Pandas is a data manipulation tool that performs operations on rows and columns along with transforming data.
- sklearn.decomposition library has been used for performing SVM and PCA which are dimentionality reduction techniques. SVM is Support Vector Machine which is used for regression and classification. PCA is Principal Component Analysis which is used to reduce the complexity of the problem.
- imblearn.over_sampling and imblearn.combine has been used for SMOTE and SMOTETomek techniques which are used for dealing with imbalanced data in the datasets.
- sklearn.metrics has been used to calculate the f1 score for the training and test data.
- import warnings and warnings.filterwarnings("ignore") has been used to ignore unnecessary warnings in the code.

Reference from text book has been taken: https://www-users.cse.umn.edu/~kumar001/dmbook/slides/chap4_ensemble.pdf





