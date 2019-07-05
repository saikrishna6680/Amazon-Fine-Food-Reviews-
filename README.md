# Amazon-Fine-Food-Reviews-Analysis-and-Modelling

Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, Tfidf,Word2Vec).
Build Several ML Models like KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, GBDT, LSTM(RNNs) etc.

**Objective:**

Given a Text Review, Determine Whether the Review is Positive (Rating of 4 or 5) or Negative (Rating of 1 or 2).

About Dataset

The Amazon Fine Food Reviews dataset consists of Reviews of Fine Foods from Amazon.

Number of Reviews: 568,454
Number of users: 256,059
Number of Products: 74,258
Timespan: Oct 1999 - Oct 2012
Number of Attributes/Columns in data: 10

Attribute Information:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review

## 1. Exploratory Data Analysis, Natural Language Processing, Text Preprocessing and Visualization using TSNE
1. Defined Problem Statement.
2. Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset Plotted Word Clouds, Distribution plots, Histograms, etc.
3. Performed Data Cleaning & Data Preprocessing(Removed html tags, Punctuations, Stopwords and Stemmed the words using Porter Stemmer).
4. Plotted TSNE with Different Perplexity values for Different Featurization like BOW(uni-gram), Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.

## 2. KNN
1. Applied K-Nearest Neighbour on Different Featurization like BOW, tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
Applying 10-fold CV by using Brute Force Algorithm to Find Optimal 'K'.
2. Calculated MissClassification Error for each K value.
3. Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix. 

## Failure Cases of K-NN:-
1. If my Query Point is Far Away from Neighbour Points I cannot Decide it's Particular Class.
2. If my Positive Points and Negative Points are jumbled so tightly there is no Useful Information in these cases my Machine learning Algorithms Fails.

## KNN Limitations:-
Knn takes large Space Complexity of order(nd) and time complexity of order(nd).

## Conclusions:
1. KNN is a very Slow Algorithm it takes very long Time to Train.
2. In K-nn We Should not take K-value even Because Classification is done by Majority vote.
2. Best Accuracy is Achieved by Avg Word2Vec Featurization Which is of 89.48%.

## 3. Naive Bayes
1. Applied Naive Bayes using Bernoulli NB and Multinomial NB on Different Featurization BOW, Tfidf.
2. Find Right Alpha(α) using Cross Validation.
3. Get Feature Importance for Positive class and Negative Class.
4. Evaluated the Test data on Various Performance metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix.

## Conclusions:
1. Naive Bayes is much Faster Algorithm than KNN.
2. Best F1 score is Acheived by Tf-idf Featurization which is 0.89.

## 4. Logistic Regression
1. Applied Logistic Regression on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
2. Find Lambda(λ) By Grid Search & Randomized Search.
3. Evaluated the Test data on various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix.
4. Showed How Sparsity Increases as we Increase lambda or decrease C when L1 Regularizer is used for each Featurization.
5. Did Pertubation Test to check whether the Features are multi-collinear or not.

## Assumptions of Logistic Regression
1) logistic Regression does not require a Linear relationship between the Dependent and Independent variables.
2) The Error Terms (residuals) do not need to be Normally Distributed.
3) Homoscedasticity is not Required.
4) Finally, the Dependent Variable in Logistic Regression is not Measured on an interval or ratio scale.

## Conclusions:
1. Sparsity Increases as we decrease C (increase lambda) When we use L1 Regularizer for Regularization.
2. Logistic Regression with Tfidf Featurization Performs best with F1_score of 0.89 and Accuracy of 93.385.
3. Logistic Regression is Faster Algorithm.

## 5. SVM
1. Applied SVM with RBF(Radial Basis Function) kernel on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
2. Find Right C and Gamma (ɣ) Using Grid Search & Randomized Search Cross Validation.
3. Applied SGDClassifier on Featurization.
3. Evaluated Test Data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix. 

## Conclusions:
1. SGD with Bow By Random Search gives Better Results.
2. SGDClasiifier Takes Very Less Time to Train.

## 6. Decision Trees
1. Applied Decision Trees on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec To find the optimal depth using Cross Validation.
2. By doing Grid Search We finded max_depth.
3. Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix.
4.Plotted wordclouds of Feature Importance of both Positive class and Negative Class.

## Conclusions:
1. Tf-idfw2vec Featurization(max_depth=11) gave the Best Results with Accuracy of 77.24.

## 7. Ensembles Models (Random Forest &Grident Boost Decision Tree)
1.Apply GBDT and Random Forest Algorithm To Find Right Baselearners using Cross Validation and Get Feature Importance for Positive
class and Negative class.
2. Performing Grid Search for getting the Best Max_depth, Learning rate.
3. Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix. 
4. Plotted Word Cloud of Feature Importance Received for RF and GBDT Classifier.

## Conclusions:
1. Avgw2vec Featurization in Random Forest (BASE-LEARNERS=120) with Grid Search gave the Best Results with F1-score of 89.0311.
2. Tfidfw2v Featurization in GBDT (Learning Rate=0.05, DEPTH=3) gave the Best Results with F1-score of 88.9755.

## 8. LSTM(RNNs)
1. Getting Vocabulary of all the words and Getting Frequency of each word, Indexing Each word Converting data into Imdb dataset format Running the lstm model and Report the Accuracy.
2. Applied Different Architectures of LSTM on Amazon Fine Food Reviews Dataset.
3. Recurrent Neural Networks(RNN) with One LSTM layer.
4. Recurrent Neural Networks(RNN) with Two LSTM layer.
5. Recurrent Neural Networks(RNN) with Three LSTM layer.
## Conclusions:
1. RNN with 1LSTM layer has got high Accuracy 92.76

