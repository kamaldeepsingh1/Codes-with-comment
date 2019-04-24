# Importing Libraries
import pandas as pd # for data manuplation
import matplotlib.pyplot as plt # for graphical purpose
import numpy as np # for numperical purpose
import seaborn as sns # for graphical 

# Importing file in workingenvironment
dataset = pd.read_csv('StudentsPerformance.csv')
dataset.info()# to know the information of dataset
# Plotting scatter graph between numerical values
pd.scatter_matrix(dataset)
corr_matrix = dataset.corr()# creating correlation matrix as heat map is plotted b/w matrix
# plooting heat map
sns.heatmap(corr_matrix, annot= True)
# ploting bar graph b/w gender and math score to get insight
dataset['gender'].value_counts()# counting values of gender column
# Finding mean of math score for gender
gender_group_mean = dataset.groupby(['gender'])['math score'].mean()
# Sorting the values
gender_group_mean = gender_group_mean.sort_values(ascending = False)
# plotting graph between index i.e. female or male and values
plt.bar(gender_group_mean.index, gender_group_mean.values, color = ['pink', 'green'])
plt.show()
# providing names to X and y axixs
plt.xlabel('gender')
plt.ylabel('math score')
plt.title('to know which gender scored more marks Maths')

# ploting bar graph b/w gender and reading score to get insight
dataset['gender'].value_counts()
gender_group_mean = dataset.groupby(['gender'])['reading score'].mean()
gender_group_mean = gender_group_mean.sort_values(ascending = False)

plt.bar(gender_group_mean.index, gender_group_mean.values, color = ['pink', 'green'])
plt.show()
plt.xlabel('gender')
plt.ylabel('reading score')
plt.title('to know which gender has more reading score')

# ploting bar graph b/w gender and writing score to get insight
dataset['gender'].value_counts()
gender_group_mean = dataset.groupby(['gender'])['writing score'].mean()
gender_group_mean = gender_group_mean.sort_values(ascending = False)

plt.bar(gender_group_mean.index, gender_group_mean.values, color = ['pink', 'green'])
plt.show()
plt.xlabel('gender')
plt.ylabel('reading score')
plt.title('to know which gender has more writing score')

# Adding math score,reading score, writing score into total mark
total = (dataset['math score']+ dataset['reading score']+dataset['writing score'])/3
dataset['total_marks'] = total # creating new column of total marks
 
#plotting graph b/w gender and total_mark
dataset['gender'].value_counts()
gender_group_mean = dataset.groupby(['gender'])['total_marks'].mean()
gender_group_mean = gender_group_mean.sort_values(ascending = False)

plt.bar(gender_group_mean.index, gender_group_mean.values, color = ['pink', 'green'])
plt.show()
plt.xlabel('gender')
plt.ylabel('total marks')
plt.title('to know which gender has more marks')

# plotting graph b/w parental level of education and total_marks to get insight
dataset['parental level of education'].value_counts()
education_group_mean = dataset.groupby(['parental level of education'])['total_marks'].mean()
education_group_mean = education_group_mean.sort_values(ascending = False)

plt.bar(education_group_mean.index, education_group_mean.values)
plt.show()
plt.legend()
plt.xlabel('education')
plt.ylabel('total marks')
plt.title('to know the marks according to education')



















