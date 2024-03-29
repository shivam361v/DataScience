#!/usr/bin/env python
# coding: utf-8

# # Activity: Explore confidence intervals

# ## Introduction

# The Air Quality Index (AQI) is the Environmental Protection Agency's index for reporting air quality. A value close to 0 signals little to no public health concern, while higher values are associated with increased risk to public health. The United States is considering a new federal policy that would create a subsidy for renewable energy in states observing an average AQI of 10 or above. <br>
# 
# You've just started your new role as a data analyst in the Strategy division of Ripple Renewable Energy (RRE). **RRE operates in the following U.S. states: `California`, `Florida`, `Michigan`, `Ohio`, `Pennsylvania`, `Texas`.** You've been tasked with constructing an analysis which identifies which of these states are most likely to be affected, should the new federal policy be enacted.

# Your manager has requested that you do the following for your analysis:
# 1. Provide a summary of the mean AQI for the states in which RRE operates.
# 2. Construct a boxplot visualization for AQI of these states using `seaborn`.
# 3. Evaluate which state(s) may be most affected by this policy, based on the data and your boxplot visualization.
# 4. Construct a confidence interval for the RRE state with the highest mean AQI.

# ## Step 1: Imports
# 
# ### Import packages
# 
# Import `pandas` and `numpy`.

# In[1]:


# Import relevant packages

### YOUR CODE HERE ###

import pandas as pd
import numpy as np


# ### Load the dataset
# 
# The dataset provided gives national Air Quality Index (AQI) measurements by state over time.  `Pandas` is used to import the file `c4_epa_air_quality.csv` as a DataFrame named `aqi`. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.
# 
# *Note: For the purposes of your analysis, you can assume this data is randomly sampled from a larger population.*

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA

### YOUR CODE HERE ###
aqi = pd.read_csv('c4_epa_air_quality.csv')


# ## Step 2: Data exploration

# ### Explore your dataset
# 
# Before proceeding to your deliverables, spend some time exploring the `aqi` DataFrame. 

# In[8]:


# Explore your DataFrame `aqi`.

### YOUR CODE HERE ###

aqi.describe(include='all')
aqi['state_name'].value_counts()


# **Question:** What time range does this data cover?
# 
# 0 - 259
#         
# 

# **Question:** What are the minimum and maximum AQI values observed in the dataset?
# 
#  0
#  

# **Question:** Are all states equally represented in the dataset?
# 
# No

# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about descriptive statisics](https://www.coursera.org/learn/the-power-of-statistics/lecture/jVRS4/measures-of-central-tendency).
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Use `pandas` or `numpy` to explore the `aqi` DataFrame.
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use any of the following functions:
# - `pandas`: `describe()`,`value_counts()`,`shape()`
# - `numpy`: `unique()`,`mean()`
#     
# </details>

# ## Step 3: Statistical tests
# 
# ### Summarize the mean AQI for RRE states
# 
# Start with your first deliverable. Summarize the mean AQI for the states in which RRE operates (California, Florida, Michigan, Ohio, Pennsylvania, and Texas).

# In[9]:


# Summarize the mean AQI for RRE states.

### YOUR CODE HERE ###

# Create a list of RRE states.

rre_states = ['California','Florida','Michigan','Ohio','Pennsylvania','Texas']

# Subset `aqi` to only consider these states.

aqi_rre = aqi[aqi['state_name'].isin(rre_states)]

# Find the mean aqi for each of the RRE states.

aqi_rre.groupby(['state_name']).agg({'aqi':'mean', 'state_name': 'count'})


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Subset your DataFrame to only include those states in which RRE operates. 
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Define a list consisting of the states in which RRE operates and use that list to subset your DataFrame. 
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Use `pandas` `isin.()` to subset your DataFrame by the list of RRE states.
#     
# </details>

# ### Construct a boxplot visualization for the AQI of these states
# 
# Seaborn is a simple visualization library, commonly imported as `sns`. Import `seaborn`. Then utilize a boxplot visualization from this library to compare the distributions of AQI scores by state.

# In[10]:


# Import seaborn as sns.

### YOUR CODE HERE ###

import seaborn as sns


# ### Create an in-line visualization showing the distribution of `aqi` by `state_name`
# 
# Now, create an in-line visualization showing the distribution of `aqi` by `state_name`.

# In[11]:


### YOUR CODE HERE ###

sns.boxplot(x=aqi_rre['state_name'], y=aqi_rre['aqi'])


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Use the boxplot visual for this purpose.
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Reference [Seaborn's boxplot visualization documentation](https://seaborn.pydata.org/generated/seaborn.boxplot.html). 
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Assign `state_name` to the x argument and `aqi` to the y.
#     
# </details>

# **Question:** Based on the data and your visualizations, which state(s) do you suspect will be most affected by this policy?

# california

# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Consider the mean AQI for the RRE states, as well as the distribution in the boxplots relative to the policy limit (10).
#     
# </details>

# ### Construct a confidence interval for the RRE state with the highest mean AQI
# 
# Recall the 4-step process in constructing a confidence interval:
# 
# 1.   Identify a sample statistic.
# 2.   Choose a confidence level.
# 3.   Find the margin of error. 
# 4.   Calculate the interval.

# ### Construct your sample statistic
# 
# To contruct your sample statistic, find the mean AQI for your state.

# In[13]:


# Find the mean aqi for your state.

### YOUR CODE HERE ###

aqi_ca = aqi[aqi['state_name'] == 'California']
sample_mean = aqi_ca['aqi'].mean()
sample_mean


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Reference what you've previously learned to recall what a [sample statistic](https://www.coursera.org/learn/the-power-of-statistics/supplement/cdOx7/construct-a-confidence-interval-for-a-small-sample-size) is.
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Calculate the mean for your highest AQI state to arrive at your sample statistic.
#     
# </details>

# <details>
#  <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Call the `mean()` function within `pandas` on your DataFrame.
#     
# </details>

# ### Choose your confidence level
# 
# Choose your confidence level for your analysis. The most typical confidence level chosen is 95%; however, you can choose 90% or 99% if you want decrease or increase (respectively) your level of confidence about your result.

# In[21]:


# Input your confidence level here:

### YOUR CODE HERE ###

cl = 0.95


# ### Find your margin of error (ME)
# 
# Recall **margin of error = z * standard error**, where z is the appropriate z-value for the given confidence level. To calculate your margin of error:
# 
# - Find your z-value. 
# - Find the approximate z for common confidence levels.
# - Calculate your **standard error** estimate. 
# 
# | Confidence Level | Z Score |
# | --- | --- |
# | 90% | 1.65 |
# | 95% | 1.96 |
# | 99% | 2.58 |
# 

# In[14]:


# Calculate your margin of error.

### YOUR CODE HERE ###

# Begin by identifying the z associated with your chosen confidence level.

z_score = 1.96

# Next, calculatemar your standard error.

standard_error = aqi_ca['aqi'].std() / np.sqrt(aqi_ca.shape[0])
# Lastly, use the preceding result to calculate your margin of error.

margin_of_error = standard_error * z_score
margin_of_error


# ### Calculate your interval
# 
# Calculate both a lower and upper limit surrounding your sample mean to create your interval.

# In[18]:


# Calculate your confidence interval (upper and lower limits).

### YOUR CODE HERE ###
upper_limit = sample_mean + margin_of_error
lower_limit = sample_mean - margin_of_error

(upper_limit, lower_limit)


# <details>
#   <summary><h4><strong>Hint 1</strong></h4></summary>
# 
# Refer to [the content about constructing a confidence interval](https://www.coursera.org/learn/the-power-of-statistics/lecture/3jbsX/construct-a-confidence-interval-for-a-proportion).
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 2</strong></h4></summary>
# 
# Identify the sample mean from your prior work. Then use the margin of error to construct your upper and lower limits.  
#     
# </details>

# <details>
#   <summary><h4><strong>Hint 3</strong></h4></summary>
# 
# Subtract the margin of error from the sample mean to construct your lower limit, and add the margin of error to your sample mean to construct your upper limit.
#     
# </details>

# ### Alternative: Construct the interval using `scipy.stats.norm.interval()`
# 
# `scipy` presents a simpler solution to developing a confidence interval. To use this, first import the `stats` module from `scipy`.

# In[23]:


# Import stats from scipy.

### YOUR CODE HERE ###

from scipy import stats


# ## Step 4: Results and evaluation
# 
# ### Recalculate your confidence interval
# 
# Provide your chosen `confidence_level`, `sample_mean`, and `standard_error` to `stats.norm.interval()` and recalculate your confidence interval.

# In[24]:


### YOUR CODE HERE ###

stats.norm.interval(alpha=cl, loc=sample_mean, scale=standard_error)


# # Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# **What findings would you share with others?**
# 
# [Write your response here. Double-click (or enter) to edit.]
# 
# **What would you convey to external stakeholders?**
# 
# [Write your response here. Double-click (or enter) to edit.]

# **References**
# 
# [seaborn.boxplot — seaborn 0.12.1 documentation](https://seaborn.pydata.org/generated/seaborn.boxplot.html). (n.d.). 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
