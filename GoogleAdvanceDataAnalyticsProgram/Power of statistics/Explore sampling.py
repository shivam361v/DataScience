#!/usr/bin/env python
# coding: utf-8

# # Activity: Explore sampling

# ## Introduction
# In this activity, you will engage in effective sampling of a dataset in order to make it easier to analyze. As a data professional you will often work with extremely large datasets, and utilizing proper sampling techniques helps you improve your efficiency in this work. 
# 
# For this activity, you are a member of an analytics team for the Environmental Protection Agency. You are assigned to analyze data on air quality with respect to carbon monoxide—a major air pollutant—and report your findings. The data utilized in this activity includes information from over 200 sites, identified by their state name, county name, city name, and local site name. You will use effective sampling within this dataset. 

# ## Step 1: Imports

# ### Import packages
# 
# Import `pandas`,  `numpy`, `matplotlib`, `statsmodels`, and `scipy`. 

# In[26]:


# Import libraries and packages

### YOUR CODE HERE ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


# ### Load the dataset
# 
# As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[27]:


# RUN THIS CELL TO IMPORT YOUR DATA.

### YOUR CODE HERE ###
epa_data = pd.read_csv("c4_epa_air_quality.csv", index_col = 0)


# <details>
#   <summary><h4>Hint 1</h4></summary>
# 
# Use the function in the `pandas` library that allows you to read in data from a csv file and load it into a DataFrame. 
# 
# </details>

# <details>
#   <summary><h4>Hint 2</h4></summary>
# 
# Use the `read_csv` function from the pandas `library`. Set the `index_col` parameter to `0` to read in the first column as an index (and to avoid `"Unnamed: 0"` appearing as a column in the resulting Dataframe).
# 
# </details>

# ## Step 2: Data exploration

# ### Examine the data
# 
# To understand how the dataset is structured, examine the first 10 rows of the data.

# In[28]:


# First 10 rows of the data

### YOUR CODE HERE ###

epa_data.head()


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the function in the `pandas` library that allows you to get a specific number of rows from the top of a DataFrame. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </STRONG></h4></summary>
# 
# Use the `head` function from the `pandas` library. Set the `n` parameter to `10` to print out the first 10 rows.
# 
# </details>

# **Question:** What does the `aqi` column represent?

# Air quality index
#         

# ### Generate a table of descriptive statistics
# 
# Generate a table of some descriptive statistics about the data. Specify that all columns of the input be included in the output.

# In[29]:


### YOUR CODE HERE ###
epa_data.describe(include='all')


# <details>
#   <summary><h4>Hint 1</h4></summary>
# 
# Use function in the `pandas` library that allows you to generate a table of basic descriptive statistics in a DataFrame.
# 
# </details>

# <details>
#   <summary><h4>Hint 2</h4></summary>
# 
# Use the `describe` function from the `pandas` library. Set the `include` parameter passed in to this function to 'all' to specify that all columns of the input be included in the output.
# 
# </details>

# **Question:** Based on the preceding table of descriptive statistics, what is the mean value of the `aqi` column? 

# 6.757

# **Question:** Based on the preceding table of descriptive statistics, what do you notice about the count value for the `aqi` column?

# 260

# ### Use the `mean()` function on the `aqi`  column
# 
# Now, use the `mean()` function on the `aqi`  column and assign the value to a variable `population_mean`. The value should be the same as the one generated by the `describe()` method in the above table. 

# In[11]:


### YOUR CODE HERE ###

population_mean = epa_data['aqi'].mean()
population_mean


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the function in the `pandas` library that allows you to generate a mean value for a column in a DataFrame.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </STRONG></h4></summary>
# 
# Use the `mean()` method.
# 
# </details>

# ## Step 3: Statistical tests

# ### Sample with replacement
# 
# First, name a new variable `sampled_data`. Then, use the `sample()` dataframe method to draw 50 samples from `epa_data`. Set `replace` equal to `'True'` to specify sampling with replacement. For `random_state`, choose an arbitrary number for random seed. Make that arbitrary number `42`.

# In[25]:


### YOUR CODE HERE ###

sample_data = epa_data.sample(n=50, replace=True, random_state=42)


# ### Output the first 10 rows
# 
# Output the first 10 rows of the DataFrame. 

# In[13]:


### YOUR CODE HERE ###
sample_data.head()


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the function in the `pandas` library that allows you to get a specific number of rows from the top of a DataFrame. 
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </STRONG></h4></summary>
# 
# Use the `head` function from the `pandas` library. Set the `n` parameter to `10` to print out the first 10 rows.
# 
# </details>

# **Question:** In the DataFrame output, why is the row index 102 repeated twice? 

# Due to replace parameter in sample function
#         

# **Question:** What does `random_state` do?

# It will pick an random instatnce from an dataset.

# ### Compute the mean value from the `aqi` column
# 
# Compute the mean value from the `aqi` column in `sampled_data` and assign the value to the variable `sample_mean`.

# In[15]:


### YOUR CODE HERE ###
sample_mean = sample_data['aqi'].mean()
sample_mean


#  **Question:**  Why is `sample_mean` different from `population_mean`?
# 

# I happen because we had chosen an sample instead of entire dataset

# ### Apply the central limit theorem
# 
# Imagine repeating the the earlier sample with replacement 10,000 times and obtaining 10,000 point estimates of the mean. In other words, imagine taking 10,000 random samples of 50 AQI values and computing the mean for each sample. According to the **central limit theorem**, the mean of a sampling distribution should be roughly equal to the population mean. Complete the following steps to compute the mean of the sampling distribution with 10,000 samples. 
# 
# * Create an empty list and assign it to a variable called `estimate_list`. 
# * Iterate through a `for` loop 10,000 times. To do this, make sure to utilize the `range()` function to generate a sequence of numbers from 0 to 9,999. 
# * In each iteration of the loop, use the `sample()` function to take a random sample (with replacement) of 50 AQI values from the population. Do not set `random_state` to a value.
# * Use the list `append()` function to add the value of the sample `mean` to each item in the list.
# 

# In[17]:


### YOUR CODE HERE ###

estimate_list = []
for i in range(1000):
    estimate_list.append(epa_data['aqi'].sample(n=50, replace=True).mean())


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Review [the content about sampling in Python](https://www.coursera.org/learn/the-power-of-statistics/lecture/SNOE0/sampling-distributions-with-python). 
# 
# </details>

# ### Create a new DataFrame
# 
# Next, create a new DataFrame from the list of 10,000 estimates. Name the new variable `estimate_df`.

# In[19]:


### YOUR CODE HERE ###
estimate_df = pd.DataFrame(data={'estimate': estimate_list})
estimate_df


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Review [the content about sampling in Python](https://www.coursera.org/learn/the-power-of-statistics/lecture/SNOE0/sampling-distributions-with-python). 
# 
# </details>

# <details>
# <summary><h4><strong> Hint 2 </STRONG></h4></summary>
# 
# Use the `mean()` function.
# 
# </details>

# ### Compute the mean() of the sampling distribution
# 
# Next, compute the `mean()` of the sampling distribution of 10,000 random samples and store the result in a new variable `mean_sample_means`.

# In[20]:


### YOUR CODE HERE ###

estimate_mean = estimate_df['estimate'].mean()
estimate_mean


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the function in the `pandas` library that allows you to generate a mean value for a column in a DataFrame.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 2 </STRONG></h4></summary>
# 
# Use the `mean()` function.
# 
# </details>

# **Question:** What is the mean for the sampling distribution of 10,000 random samples?

# 6.774 is the mean of above distribution

# <details>
#   <summary><h4><strong> Hint 3 </STRONG></h4></summary>
# 
# This value is contained in `mean_sample_means`.
# 
# </details>

# <details>
#   <summary><h4><strong> Hint 4 </STRONG></h4></summary>
# 
# According to the central limit theorem, the mean of the preceding sampling distribution should be roughly equal to the population mean. 
# 
# </details>

# **Question:** How are the central limit theorem and random sampling (with replacement) related?

# The sample distribution of random sampling is normal distribution and central limit theorem is also an normal distribution.

# ### Output the distribution using a histogram
# 
# Output the distribution of these estimates using a histogram. This provides an idea of the sampling distribution.

# In[21]:


### YOUR CODE HERE ###

estimate_df['estimate'].hist()


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the `hist()` function. 
# 
# </details>

# ### Calculate the standard error
# 
# Calculate the standard error of the mean AQI using the initial sample of 50. The **standard error** of a statistic measures the sample-to-sample variability of the sample statistic. It provides a numerical measure of sampling variability and answers the question: How far is a statistic based on one particular sample from the actual value of the statistic?

# In[31]:


### YOUR CODE HERE ###

standard_error  = sample_data['aqi'].std() / np.sqrt(len(sample_data))
standard_error


# <details>
#   <summary><h4><strong> Hint 1 </STRONG></h4></summary>
# 
# Use the `std()` function and the `np.sqrt()` function.
# 
# </details>

# ## Step 4: Results and evaluation

# ###  Visualize the relationship between the sampling and normal distributions
# 
# Visualize the relationship between your sampling distribution of 10,000 estimates and the normal distribution.
# 
# 1. Plot a histogram of the 10,000 sample means 
# 2. Add a vertical line indicating the mean of the first single sample of 50
# 3. Add another vertical line indicating the mean of the means of the 10,000 samples 
# 4. Add a third vertical line indicating the mean of the actual population

# In[33]:


### YOUE CODE HERE ###
   
plt.figure(figsize=(8,5))
plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, population_mean, standard_error)
plt.plot(x, p, 'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=population_mean, color='m', linestyle = 'solid', label = 'population mean')
plt.axvline(x=sample_mean, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("Sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1));


# **Question:** What insights did you gain from the preceding sampling distribution?

# [Write your response here. Double-click (or enter) to edit.]

# # Considerations
# 
# **What are some key takeaways that you learned from this lab?**
# 
# **What findings would you share with others?**
# 
# **What would you convey to external stakeholders?**
# 
# 
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
