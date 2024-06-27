#!/usr/bin/env python
# coding: utf-8

# # Binomial logistic regression (Part 1)
# 

# Throughout the following exercises, you will learn to use Python to build and evaluate a binomial logistic regression model. Before starting on this programming exercise, we strongly recommend watching the video lecture and completing the IVQ for the associated topics.

# All the information you need for solving this assignment is in this notebook, and all the code you will be implementing will take place within this notebook.

# As we move forward, you can find instructions on how to install required libraries as they arise in this notebook. Before we begin with the exercises and analyzing the data, we need to import all libraries and extensions required for this programming exercise. Throughout the course, we will be using pandas and sickit-learn for operations, and seaborn for plotting.

# ## Relevant imports

# Begin by importing the relevant packages and data.

# In[1]:


# Import pandas and seaborn packages
import pandas as pd
import seaborn as sns


# ## Exploratory data analysis 

# **Note:** The following code cell is shown in the video, but it will only work if the .csv file is in the same folder as the notebook. Otherwise, please follow the data loading process outlined above.

# In[3]:


# Load in if csv file is in the same folder as notebook
activity = pd.read_csv("activity.csv")


# In[4]:


# Get summary statistics about the dataset
activity.describe()


# In[5]:


# Examine the dataset
activity.head()


# ## Construct binomial logistic regression model

# For binomial logistic regression, we'll be using the `scikit-learn` package, which is frequently used for machine learning and more advanced data science topics. For the purposes of this exercise, we'll only load in the functions we need: `train_test_split()` and `LogisticRegression()`.

# In[6]:


# Load in sci-kit learn functions for constructing logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Then, we'll save the data into variables called X and y so we can use the `train_test_split()` function more easily. Remember that you can subset specific columns of a DataFrame object by using double square brackets: `[[]]` and listing the columns in between, separated by commas.

# In[7]:


# Save X and y data into variables
X = activity[["Acc (vertical)"]]
y = activity[["LyingDown"]]


# Then we'll split the data into training and holdout datasets. We set the `test_size` to `0.3` so that the holdout dataset is only 30% of the total data we have. We'll set the `random_state` equal to `42`. If you change this variable, then your results will be different from ours. Setting the `random_state` is mainly for reproducibility purposes.

# In[11]:


# Split dataset into training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=41)


# Then we'll build our classifier, and fit the model to the data by using the `.fit()` function. We'll save the fitted model as a variable called `clf`.

# In[12]:


clf = LogisticRegression().fit(X_train,y_train)


# ## Get coefficients and visualize model

# We can use the `coef_` and `intercept_` attributes of the `clf` object to get the coefficient and intercept of our model.

# In[13]:


# Print the coefficient
clf.coef_


# In[14]:


# Print the intercept
clf.intercept_


# So, based on what we've found, our model has an intercept or $\beta_0$ of 6.10 and a $\beta_1$ of -0.12. Now we can plot our model and data with a 95% confidence band using the `regplot()` function from the `seaborn` package. Remember to set the argument `logistic=True` so that the function knows we are plotting a logistic regression model, not a linear regression model.

# In[15]:


# Plot the logistic regression and its confidence band
sns.regplot(x="Acc (vertical)", y="LyingDown", data=activity, logistic=True)


# # Confusion matrix (Part II)

# This part of the notebook contains all of the code that will be presented in the second part of this section in the course. The focus is on **confusion matrices**, which are used to evaluate classification models, such as a binomial logistic regression model. 
# 
# **Note:** We are assuming that the earlier parts of this notebook have been run, and that the existing variables and imported packages have been saved. 

# ## Construct logistic regression model

# Once again, we split our data, which is currently saved as variables `X` and `y`, into training and holdout datasets using the `train_test_split()` function. The function has already been imported from the `scikit-learn` package. Then, we build the model by using the `LogisticRegression()` function with the `.fit()` function.
# 
# Next, we can save our model's predictions by inputting the holdout sample, `X_test` into the model's `.predict()` function.

# In[16]:


# Split data into training and holdout samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build regression model
clf = LogisticRegression().fit(X_train,y_train)

# Save predictions
y_pred = clf.predict(X_test)


# We can print out the predicted labels by just calling on `clf.predict(X_test)`. Recall that 0 means not lying down, and 1 means lying down.

# In[17]:


# Print out the predicted labels
clf.predict(X_test)


# But, the model actually calculates a probability that given a particular value of X, the person is lying down. We can print out the predicted probabilities with the following line of code. You can read more about the [`LogisticRegression()` function](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), its attributes, and related functions on the `scikit-learn` website.

# In[18]:


# Print out the predicted probabilities
clf.predict_proba(X_test)[::,-1]


# ## Create confusion matrix
# 
# 

# To finish this part of the course, we'll create a confusion matrix. Recall the following definition:
# 
# * **Confusion matrix:** A graphical representation of how accurate a classifier is at predicting the labels for a categorical variable.
# 
# To create a confusion matrix, we'll use the [`confusion_matrix()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html?highlight=confusion_matrix#sklearn.metrics.confusion_matrix) function from the `metrics` module of `scikit-learn`. To use the function, we'll need to input the following:
# * Actual labels of the holdout sample, stored as `y_test`
# * Predicted labels of the holdout sample, stored as `y_pred`
# * The names of the labels, which you can access using `clf.classes_`
# 
# **Note:** If there were more classes, we would have more numbers or labels in `clf.classes_`. Since this is a binomial logistic regression, there are only two labels, 0 and 1.

# In[19]:


# Import the metrics module from scikit-learn
import sklearn.metrics as metrics


# In[20]:


# Calculate the values for each quadrant in the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)


# In[21]:


# Create the confusion matrix as a visualization
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)


# In order to understand and interpret the numbers in the below confusion matrix, it is important to keep the following in mind:
# 
# * The upper-left quadrant displays the number of **true negatives**.
# * The bottom-left quadrant displays the number of **false negatives**.
# * The upper-right quadrant displays the number of **false positives**.
# * The bottom-right quadrant displays the number of **true positives**.
# 
# We can define the above bolded terms as follows in our given context:
# * **True negatives**: The number of people that were not lying down that the model accurately predicted were not lying down.
# * **False negatives**: The number of people that were lying down that the model inaccurately predicted were not lying down.
# * **False positives**: The number of people that were not lying down that the model inaccurately predicted were lying down.
# * **True positives**: The number of people that were lying down that the model accurately predicted were lying down.
# 
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.

# In[22]:


# Display the confusion matrix
disp.plot()


# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
# 
# You now understand how to build and evaluate a binomial logistic regression model with Python. Going forward, you can start using binomial logistic regression models with your own datasets.
