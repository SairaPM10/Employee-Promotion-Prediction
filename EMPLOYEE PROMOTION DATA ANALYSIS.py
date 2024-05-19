#!/usr/bin/env python
# coding: utf-8

# **EMPLOYEE PROMOTION PREDICTION: CLUSTERING, CLASSIFICATION AND PREDICTION**
# 

# In[2]:


import pandas as pd

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/800train.csv'
data = pd.read_csv(file_path)

# Columns to check for missing values
columns_to_check = ['department', 'education', 'previous_year_rating', 
                    'length_of_service', 'awards_won?', 'avg_training_score', 'is_promoted']

# Checking for missing values in the specified columns
missing_values = data[columns_to_check].isnull().sum()


print(missing_values)



# **DATA PRE-PROCESSING**

# In[3]:


# Treating missing values
# For numerical columns, use median
# For categorical columns, use mode

# Identifying categorical and numerical columns
categorical_columns = ['department', 'education']
numerical_columns = ['previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score', 'is_promoted']

# Filling missing values for numerical columns with median
for col in numerical_columns:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)

# Filling missing values for categorical columns with mode
for col in categorical_columns:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)

# Converting categorical columns to numerical
# Using one-hot encoding for nominal data
data = pd.get_dummies(data, columns=categorical_columns)

# Checking the dataset after treatment and conversion
data.head()


# In[6]:


from sklearn.preprocessing import LabelEncoder

# Re-load the original dataset to ensure a fresh start
data_original = pd.read_csv(file_path)

# Specifying the columns to be included
included_columns = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
                    'awards_won?', 'avg_training_score', 'gender', 'department', 'education']

# Creating a new dataframe with the specified columns
data_for_processing = data_original[included_columns]

# Treating missing values for numerical columns with median
numerical_columns_for_processing = ['no_of_trainings', 'age', 'previous_year_rating', 
                                    'length_of_service', 'awards_won?', 'avg_training_score']
for col in numerical_columns_for_processing:
    median_value = data_for_processing[col].median()
    data_for_processing[col].fillna(median_value, inplace=True)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Converting 'department' and 'education' to numerical
data_for_processing['department_new'] = label_encoder.fit_transform(data_for_processing['department'])
data_for_processing['education_new'] = label_encoder.fit_transform(data_for_processing['education'])

# Converting 'gender' to numerical
data_for_processing['gender_new'] = label_encoder.fit_transform(data_for_processing['gender'])

# Removing the original 'department', 'education', and 'gender' columns
data_for_processing.drop(['department', 'education', 'gender'], axis=1, inplace=True)

# Checking for missing values again
missing_values_after_processing = data_for_processing.isnull().sum()

missing_values_after_processing, data_for_processing.head()


# In[10]:


# Modified function to calculate descriptive statistics using list and concat
def calculate_descriptive_statistics(df):
    stats_list = []

    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            stats = {
                'Variable': column,
                'N': df[column].count(),
                'N*': df[column].isna().sum(),
                'Mean': df[column].mean(),
                'StDev': df[column].std(),
                'Variance': df[column].var(),
                'Minimum': df[column].min(),
                'Q1': df[column].quantile(0.25),
                'Median': df[column].median(),
                'Q3': df[column].quantile(0.75),
                'Maximum': df[column].max(),
                'Range': df[column].max() - df[column].min(),
                'IQR': df[column].quantile(0.75) - df[column].quantile(0.25),
                'Mode': df[column].mode()[0],
                'N for Mode': df[df[column] == df[column].mode()[0]][column].count(),
                'Skewness': df[column].skew(),
                'Kurtosis': df[column].kurtosis()
            }
            stats_list.append(stats)

    return pd.DataFrame(stats_list)

# Recalculating the descriptive statistics for the trimmed dataset
descriptive_stats_trimmed = calculate_descriptive_statistics(data_trimmed)

# Formatting the output as a table
descriptive_stats_trimmed_table = descriptive_stats_trimmed.set_index('Variable')
descriptive_stats_trimmed_table


# **DATA VISUALISATION**

# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Decode 'deptnew' back to 'department' for meaningful labels
# This dictionary is based on the encoding shown previously in the descriptive statistics
department_mapping = {
    0: 'Analytics',
    1: 'Finance',
    2: 'HR',
    3: 'Legal',
    4: 'Operations',
    5: 'Procurement',
    6: 'R&D',
    7: 'Sales & Marketing',
    8: 'Technology'
}
data_trimmed['department'] = data_trimmed['department_new'].map(department_mapping)

# Plotting a 3D bar graph for the 'department' column
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Count the frequency of each department
dept_counts = data_trimmed['department'].value_counts().sort_index()
xpos = range(len(dept_counts))
ypos = [0] * len(dept_counts)
zpos = [0] * len(dept_counts)
dx = dy = [0.8] * len(dept_counts)
dz = dept_counts.values

ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

# Labeling
ax.set_xticks(xpos)
ax.set_xticklabels(dept_counts.index, rotation=90)
ax.set_xlabel('Department')
ax.set_ylabel('Frequency')
ax.set_zlabel('Count')

# Showing the 3D bar plot
plt.show()


# In[17]:


# Checking if 'is_promoted' column is in the data_trimmed DataFrame
is_promoted_included = 'is_promoted' in data_trimmed.columns

# If 'is_promoted' is not in data_trimmed, add it from the original dataset
if not is_promoted_included:
    # Add 'is_promoted' column from the original dataset
    data_trimmed['is_promoted'] = data_original['is_promoted'].head(800)

    

is_promoted_included, data_trimmed.columns.tolist()  # Show columns to verify 'is_promoted' is included


# In[19]:


# Creating a colorful pie chart for 'education'
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig, ax = plt.subplots()
education_counts = data_trimmed['education_new'].value_counts()
ax.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('Education Distribution')
plt.show()

# Creating a different visualization for 'is_promoted', using a bar chart
fig, ax = plt.subplots()
promotion_counts = data_trimmed['is_promoted'].value_counts()
promotion_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
ax.set_title('Promotion Status Counts')
ax.set_ylabel('Frequency')
ax.set_xlabel('Promotion Status')
plt.show()


# In[21]:


# Creating an improved pie chart for 'education' with better readability
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']  # Colorful scheme
explode = (0.1, 0.1, 0.1, 0.1)  # 'Explode' each slice slightly from the center

# Increase figure size for better readability
plt.figure(figsize=(8, 6))
education_counts = data_trimmed['education_new'].value_counts()
plt.pie(education_counts, labels=education_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors, explode=explode, pctdistance=0.85)

# Draw a circle at the center to turn the pie into a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.title('Education Distribution')
plt.show()


# In[22]:


# Since 'awards_won?' is a binary categorical variable, we will use a bar chart to visualize it.
awards_won_counts = data_trimmed['awards_won?'].value_counts()

# Creating the bar chart
plt.figure(figsize=(8, 6))
awards_won_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Awards Won Distribution')
plt.xlabel('Awards Won')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['Not Won', 'Won'], rotation=0)  # Assuming 0 is 'Not Won', 1 is 'Won'
plt.show()


# In[23]:


import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Data for the visualizations
variables = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 
             'avg_training_score', 'department_new', 'education_new', 'gender_new']
data_visual = data_trimmed[variables]

# Create a figure with subplots
fig, axes = plt.subplots(4, 2, figsize=(15, 20))

# Bar graphs for the 'Mode' of each variable
for i, var in enumerate(variables):
    # Skip the 'awards_won?' column for visualization
    sns.countplot(x=var, data=data_visual, ax=axes[i//2, i%2], palette='viridis')

# Adjust layout for better fit and display the plot
plt.tight_layout()
plt.show()

# Skewness in a bar graph
plt.figure(figsize=(10, 6))
sns.barplot(x=variables, y=data_visual.skew(), palette='coolwarm')
plt.title('Skewness of Variables')
plt.xticks(rotation=45)
plt.show()

# Box plots for the distribution of each variable
fig, axes = plt.subplots(4, 2, figsize=(15, 20))

for i, var in enumerate(variables):
    # Skip the 'awards_won?' column for visualization
    sns.boxplot(y=var, data=data_visual, ax=axes[i//2, i%2], palette='mako')

# Adjust layout for better fit and display the plot
plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
# Bar graph for 'no_of_trainings'
trainings_counts = data_trimmed['no_of_trainings'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
trainings_counts.plot(kind='bar', color=plt.cm.Paired(np.arange(len(trainings_counts))))
plt.title('Number of Trainings Frequency')
plt.xlabel('Number of Trainings')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()


# In[26]:


# Histogram for 'age'
plt.figure(figsize=(10, 6))
plt.hist(data_trimmed['age'], bins=range(int(data_trimmed['age'].min()), int(data_trimmed['age'].max()) + 1, 1), 
         color='skyblue', edgecolor='black')
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[29]:


# Pie chart for 'previous_year_rating'
ratings_counts = data_trimmed['previous_year_rating'].value_counts()

# Colorful scheme for the pie chart
colors = plt.cm.Paired(range(len(ratings_counts)))

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(ratings_counts, labels=ratings_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Distribution of Previous Year Rating')
plt.show()



# In[30]:


# Box plot for 'avg_training_score'
plt.figure(figsize=(10, 6))
sns.boxplot(x=data_trimmed['avg_training_score'], color='lightblue')
plt.title('Box Plot of Average Training Score')
plt.xlabel('Average Training Score')
plt.show()


# In[34]:


# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
cleaned_data = pd.read_csv(file_path)

# Assuming the categorical columns have been removed or transformed to numerical values,
# we will calculate the descriptive statistics only for the numerical columns.
# Let's identify numerical columns first (excluding the ID column if present).
numerical_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns

# Calculate descriptive statistics for these columns
desc_stats = cleaned_data[numerical_cols].describe().round(2)

# Add additional statistics: variance, range, IQR, mode, N for Mode, skewness, kurtosis
desc_stats.loc['variance'] = cleaned_data[numerical_cols].var().round(2)
desc_stats.loc['range'] = desc_stats.loc['max'] - desc_stats.loc['min']
desc_stats.loc['IQR'] = desc_stats.loc['75%'] - desc_stats.loc['25%']
modes = cleaned_data[numerical_cols].mode().iloc[0].round(2)
desc_stats.loc['mode'] = modes
desc_stats.loc['N for Mode'] = cleaned_data[numerical_cols].apply(lambda x: (x == modes[x.name]).sum())
desc_stats.loc['Skewness'] = cleaned_data[numerical_cols].skew().round(2)
desc_stats.loc['Kurtosis'] = cleaned_data[numerical_cols].kurtosis().round(2)

# Transpose the DataFrame to get variables as rows
desc_stats = desc_stats.T

# Filter out the descriptive statistics for the requested stats
required_stats = ['count', 'mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max', 'range', 'IQR', 'mode', 'N for Mode', 'Skewness', 'Kurtosis']
desc_stats = desc_stats[required_stats]

# Rename the indices and columns to match the required output
desc_stats.index.name = 'Variable'
desc_stats = desc_stats.rename(columns={'50%': 'Median', '25%': 'Q1', '75%': 'Q3', 'std': 'StDev', 'count': 'N'})
desc_stats.reset_index(inplace=True)

# Now let's display the descriptive statistics as requested.
desc_stats


# In[35]:


# Since the Python execution environment is experiencing issues, let's manually calculate and create a table for the descriptive statistics.

# First, we define the data based on the image provided for the required numerical columns.
data = {
    "Variable": ["no_of_trainings", "age", "previous_year_rating", "length_of_service", "avg_training_score"],
    "N": [800, 800, 800, 800, 800],
    "Mean": [1.25, 34.64, 3.34, 5.75, 63.87],
    "StDev": [0.65, 7.49, 1.21, 4.30, 13.48],
    "Variance": [0.42, 56.17, 1.46, 18.49, 181.77],
    "Minimum": [1.0, 20.0, 1.0, 1.0, 39.0],
    "Q1": [1.0, 29.0, 3.0, 3.0, 52.0],
    "Median": [1.0, 33.0, 3.0, 5.0, 60.0],
    "Q3": [1.0, 39.0, 4.0, 7.0, 77.0],
    "Maximum": [7.0, 60.0, 5.0, 28.0, 99.0],
    "Range": [6.0, 40.0, 4.0, 27.0, 60.0],
    "IQR": [0.0, 10.0, 1.0, 4.0, 25.0],
    "Mode": [1.0, 30.0, 3.0, 4.0, 50.0],
    "N for Mode": [649, 64, 326, 117, 42],
    "Skewness": [4.03, 0.98, -0.27, 1.72, 0.41],
    "Kurtosis": [23.32, 0.88, -0.59, 3.87, -1.06]
}

# Convert to pandas DataFrame
descriptive_stats_df = pd.DataFrame(data)

# Display the DataFrame
descriptive_stats_df


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

# Since the Python environment is not currently executing code,
# I will provide you with a sample code that you can run in your local environment.

# Sample code to create a histogram with a line indicating skewness for 'length_of_service'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Create a histogram for 'length_of_service'
plt.figure(figsize=(10, 6))
sns.histplot(df['length_of_service'], kde=True, color='skyblue')
plt.title('Length of Service Distribution with Skewness Indicator')
plt.axvline(df['length_of_service'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(df['length_of_service'].median(), color='green', linestyle='dashed', linewidth=2, label='Median')

# Annotate skewness on the plot
skewness = df['length_of_service'].skew()
plt.text(x=df['length_of_service'].max(), y=20, s=f'Skewness: {skewness:.2f}', color='red')

plt.xlabel('Length of Service (Years)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Reminder: Replace 'path_to_your_cleaned_data.csv' with the actual path to your dataset file.
# The code assumes 'length_of_service' is the correct column name in your dataset. 
# If the column name differs, please adjust it accordingly.


# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Create a box plot for 'length_of_service'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['length_of_service'], color='lightgreen')
plt.title('Box Plot of Length of Service')
plt.xlabel('Length of Service (Years)')
plt.show()


# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# Create a histogram with a KDE overlay for the 'age' column
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')

# Add titles and labels
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# **CORRELATION MATRIX**

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Compute the correlation matrix
corr_matrix = df.corr()

# Generate a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='PuBuGn', square=True, cbar=True)
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  # Adjust the layout to fit all labels
plt.show()


# In[53]:


data = pd.read_csv ('/Users/sairamoharana/Downloads/cleaned_data.csv')
data


# **CLUSTERING**
# 

# In[54]:


# Select the numeric columns for clustering
numeric_data = data[['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score','awards_won?']]


# In[55]:


# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numeric_data)
data_scaled


# In[56]:


# Determine the optimal number of clusters (K)
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-cluster Sum of Squares)')
plt.show()


# In[ ]:


# Choose the optimal number of clusters (K)
k = 3  # You can choose based on the elbow plot

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)
# Print the cluster of each "Branch ID"
for index, row in data.iterrows():
    print(f"Branch ID: {row['Branch ID']} is in Cluster {row['Cluster']}")


# In[57]:


# Choose the optimal number of clusters (K)
k = 5  # This value is chosen based on prior analysis such as the elbow method or silhouette scores

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Print the cluster of each employee using the DataFrame index as an identifier
for index, cluster in enumerate(data['Cluster']):
    print(f"Employee at index {index} is in Cluster {cluster}")


# In[69]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Standardizing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('is_promoted', axis=1))  # Excluding the target variable

# Function to compute clustering metrics for different values of k
def clustering_metrics(data, k_values):
    results = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        
        silhouette_avg = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        
        results.append([k, silhouette_avg, davies_bouldin])
    
    return pd.DataFrame(results, columns=['k', 'Silhouette Score', 'Davies-Bouldin Score'])

# Compute metrics for k = 2, 3, 4, 5
k_values = [2, 3, 4, 5]
metrics_table = clustering_metrics(scaled_data, k_values)
metrics_table


# In[70]:


# Function to perform K-Means clustering and plot the results with centroids for a given k
def kmeans_clustering_and_plot(data, k, feature_names):
    # Extracting the columns for the specified features
    feature_indices = [data.columns.get_loc(name) for name in feature_names]
    selected_data = data.iloc[:, feature_indices]

    # Applying K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(selected_data)

    # Creating a DataFrame with labels and the chosen features
    selected_data['Cluster'] = labels

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(selected_data.iloc[:, 0], selected_data.iloc[:, 1], c=labels, cmap='viridis', marker='o', label='Data Points')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()

# Visualizing K-Means clustering with centroids for k=3
kmeans_clustering_and_plot(data, k=3, feature_names=['age', 'avg_training_score'])

# Visualizing K-Means clustering with centroids for k=5
kmeans_clustering_and_plot(data, k=5, feature_names=['age', 'avg_training_score'])


# In[72]:


import numpy as np  # Importing numpy for random selection

# Creating a linkage matrix using a subset of the data to avoid a messy dendrogram
# Selecting a random subset of 50 data points for the dendrogram to make it clearer
subset_indices = np.random.choice(range(len(df_scaled)), size=50, replace=False)
df_subset = df_scaled[subset_indices]

# Creating the linkage matrix
linkage_matrix = linkage(df_subset, method='ward')

# Plotting the dendrogram for the subset
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram (Subset of 50)')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()


# **CLASSIFICATION**

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
df = pd.read_csv(file_path)

# Separating the features and the target variable
X = df.drop('is_promoted', axis=1)
y = df['is_promoted']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Checking the split
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predict on the test set
y_pred_dt = decision_tree.predict(X_test)

# Generate confusion matrix for Decision Tree Classifier
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Promoted', 'Promoted'], yticklabels=['Not Promoted', 'Promoted'])
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report for Decision Tree
print("Classification Report for Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear')

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = svm_classifier.predict(X_test)

# Generate the confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display the confusion matrix using Seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SVM Classifier')
plt.show()

# Display the classification report
print("Classification Report for SVM Classifier:")
print(class_report)


# In[5]:


from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import numpy as np

# We need to retrain the classifiers since the session was reset. 
# First, let's set up our environment and load the data again.
# Loading the data
data = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Preprocessing
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']
X_scaled = scaler.fit_transform(X)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Training SVM
svm_classifier = SVC(kernel='linear', probability=True)  # probability=True to get predict_proba required for ROC
svm_classifier.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]  # probabilities for the positive class
y_pred_proba_svm = svm_classifier.predict_proba(X_test)[:, 1]  # probabilities for the positive class

# Compute ROC curve and ROC area for each class
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curve for both classifiers
plt.figure(figsize=(10, 8))
lw = 2
plt.plot(fpr_dt, tpr_dt, color='orange', lw=lw, label='Decision Tree ROC curve (area = %0.2f)' % roc_auc_dt)
plt.plot(fpr_svm, tpr_svm, color='blue', lw=lw, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[6]:


# Re-importing necessary libraries as the code execution state was reset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Step 2: Logistic Regression

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
report_log_reg = classification_report(y_test, y_pred_log_reg)

accuracy_log_reg, roc_auc_log_reg, report_log_reg


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_proba_log_reg = log_reg.predict_proba(X_test)[:, 1]

# Evaluate the Logistic Regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)
report_log_reg = classification_report(y_test, y_pred_log_reg)

# Initialize the Lasso Regression model
lasso_reg = Lasso(alpha=0.01)  # alpha is the regularization strength

# Train the model
lasso_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_lasso_reg = lasso_reg.predict(X_test)

# Since Lasso is a regression model, we need to apply a threshold to convert predictions to binary classification
y_pred_lasso_class = (y_pred_lasso_reg >= 0.5).astype(int)

# Evaluate the Lasso Regression model
accuracy_lasso_reg = accuracy_score(y_test, y_pred_lasso_class)
report_lasso_reg = classification_report(y_test, y_pred_lasso_class)

# Display the results for Logistic Regression
print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_log_reg}")
print(f"ROC-AUC Score: {roc_auc_log_reg}")
print("Classification Report:")
print(report_log_reg)

# Display the results for Lasso Regression
print("\nLasso Regression Results:")
print(f"Accuracy: {accuracy_lasso_reg}")
print("Classification Report:")
print(report_lasso_reg)


# In[9]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Since the session has been reset, we need to preprocess the data again
# Load the dataset
data = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Preprocess the data
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier on a subset of the data for visualization purposes
# Using a max_depth to limit the complexity of the tree
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=["Not Promoted", "Promoted"], rounded=True)
plt.title("Decision Tree Classifier")
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Lasso Regression
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, y_train)
y_pred_lasso_reg = lasso_reg.predict(X_test)

# Metrics
mse_log_reg = mean_squared_error(y_test, y_pred_log_reg)
r2_log_reg = r2_score(y_test, y_pred_log_reg)
ssr_log_reg = np.sum((y_pred_log_reg - np.mean(y_test))**2)
sse_log_reg = np.sum((y_test - y_pred_log_reg)**2)
sst_log_reg = np.sum((y_test - np.mean(y_test))**2)

mse_lasso_reg = mean_squared_error(y_test, y_pred_lasso_reg)
r2_lasso_reg = r2_score(y_test, y_pred_lasso_reg)
ssr_lasso_reg = np.sum((y_pred_lasso_reg - np.mean(y_test))**2)
sse_lasso_reg = np.sum((y_test - y_pred_lasso_reg)**2)
sst_lasso_reg = np.sum((y_test - np.mean(y_test))**2)

(mse_log_reg, r2_log_reg, ssr_log_reg, sse_log_reg, sst_log_reg), (mse_lasso_reg, r2_lasso_reg, ssr_lasso_reg, sse_lasso_reg, sst_lasso_reg)


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Lasso Regression
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(X_train, y_train)
y_pred_lasso_reg = lasso_reg.predict(X_test)

# Metrics for Logistic Regression
mse_log_reg = mean_squared_error(y_test, y_pred_log_reg)
r2_log_reg = r2_score(y_test, y_pred_log_reg)
ssr_log_reg = np.sum((y_pred_log_reg - np.mean(y_test))**2)
sse_log_reg = np.sum((y_test - y_pred_log_reg)**2)
sst_log_reg = np.sum((y_test - np.mean(y_test))**2)

# Metrics for Lasso Regression
mse_lasso_reg = mean_squared_error(y_test, y_pred_lasso_reg)
r2_lasso_reg = r2_score(y_test, y_pred_lasso_reg)
ssr_lasso_reg = np.sum((y_pred_lasso_reg - np.mean(y_test))**2)
sse_lasso_reg = np.sum((y_test - y_pred_lasso_reg)**2)
sst_lasso_reg = np.sum((y_test - np.mean(y_test))**2)

# Displaying results in a neat format
results = {
    "Model": ["Logistic Regression", "Lasso Regression"],
    "MSE": [mse_log_reg, mse_lasso_reg],
    "R2": [r2_log_reg, r2_lasso_reg],
    "SSR": [ssr_log_reg, ssr_lasso_reg],
    "SSE": [sse_log_reg, sse_lasso_reg],
    "SST": [sst_log_reg, sst_lasso_reg]
}

results_df = pd.DataFrame(results)
results_df


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test)
y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
report_rf = classification_report(y_test, y_pred_rf)

# Display the results
print(f'Accuracy: {accuracy_rf}')
print(f'ROC-AUC Score: {roc_auc_rf}')
print('Classification Report:')
print(report_rf)


# In[1]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your data
data = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')

# Preprocess the data
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier
# Using class_weight='balanced' to adjust weights inversely proportional to class frequencies
svm_classifier = SVC(kernel='linear', class_weight='balanced', probability=True)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm_classifier.predict(X_test)
y_pred_proba_svm = svm_classifier.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_proba_svm)
report_svm = classification_report(y_test, y_pred_svm)

# Print the results
print(f'Accuracy: {accuracy_svm}')
print(f'ROC-AUC Score: {roc_auc_svm}')
print('Classification Report:')
print(report_svm)


# In[5]:


import pandas as pd

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
data.head()


# In[6]:


from sklearn.model_selection import train_test_split

# Splitting the dataset into features and target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Classification report
print("Decision Tree Classifier Report:")
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:")
print(cm_dt)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, dt_classifier.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.show()


# In[8]:


from sklearn.svm import SVC

# Initialize the SVM classifier
svm_classifier = SVC(probability=True)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_classifier.predict(X_test)

# Classification report
print("Support Vector Machine Classifier Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix:")
print(cm_svm)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, svm_classifier.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc="lower right")
plt.show()


# In[7]:


# Since the previous environment state has been reset, let's start by re-importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Re-importing and preprocessing the data
data = pd.read_csv('/Users/sairamoharana/Downloads/cleaned_data.csv')
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Training Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)

# Generating confusion matrix and classification report
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)
class_report_gb = classification_report(y_test, y_pred_gb, target_names=['Not Promoted', 'Promoted'], output_dict=True)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_gb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Converting classification report to DataFrame for a neat print
class_report_gb_df = pd.DataFrame(class_report_gb).transpose()
class_report_gb_df


# In[9]:


# Re-importing necessary libraries and data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_classifier.predict(X_test)

# Create the confusion matrix and classification report
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=['Not Promoted', 'Promoted'])

# Print the classification report for Random Forest Classifier
print("Classification Report for Random Forest Classifier:")
print(report_rf)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



# In[10]:


# Re-importing necessary libraries and data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate features and target
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set and calculate probabilities
y_pred_rf = rf_classifier.predict(X_test)
y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]

# Create the confusion matrix and classification report
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf, target_names=['Not Promoted', 'Promoted'])

# Calculate ROC AUC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()

# Output the AUC score and a short interpretation
interpretation = (f"The Random Forest classifier achieves an AUC of {roc_auc_rf:.2f}, "
                  "indicating a good ability to distinguish between employees who will be promoted and those who won't. "
                  "With high precision for 'Not Promoted' and moderate recall for 'Promoted', "
                  "the model effectively identifies non-promotable cases while being somewhat cautious in predicting promotions.")
roc_auc_rf, interpretation


# In[11]:


# Now let's calculate the ROC AUC for Gradient Boosting Classifier and plot the ROC curve.

# Initialize and train the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)

# Predict on the test set and calculate probabilities
y_pred_gb = gb_classifier.predict(X_test)
y_pred_proba_gb = gb_classifier.predict_proba(X_test)[:, 1]

# Calculate ROC AUC for Gradient Boosting
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_proba_gb)
roc_auc_gb = roc_auc_score(y_test, y_pred_proba_gb)

# Plot ROC curve for both Random Forest and Gradient Boosting
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve for RF (area = {roc_auc_rf:.2f})')
plt.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'ROC curve for GB (area = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

roc_auc_gb


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Assuming you have a DataFrame 'data' with features and 'is_promoted' as the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=0)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)

# Confusion Matrix for Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Creating a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix Heatmap - Gradient Boosting')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()



# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Your confusion matrix values for Random Forest Classifier
cm_rf = [[141, 4], [11, 4]]  # replace with your actual confusion matrix values

# Create the heatmap for Random Forest confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap='Blues', xticklabels=["Not Promoted", "Promoted"], yticklabels=["Not Promoted", "Promoted"])
plt.title('Confusion Matrix Heatmap for Random Forest Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Show the heatmap
plt.show()


# In[21]:


pip install -U scikit-learn


# In[11]:


from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score

file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Assuming 'data' is your DataFrame and 'is_promoted' is the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Engineering: Adding polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Re-initializing and training the models with the transformed data
log_reg_poly = LogisticRegression()
ridge_reg_poly = Ridge()

log_reg_poly.fit(X_train_poly, y_train)
ridge_reg_poly.fit(X_train_poly, y_train)

# Making predictions with the new models
y_pred_log_poly = log_reg_poly.predict(X_test_poly)
y_pred_ridge_poly = ridge_reg_poly.predict(X_test_poly)

# Evaluating the models
log_report_poly = classification_report(y_test, y_pred_log_poly)
ridge_mse_poly = mean_squared_error(y_test, y_pred_ridge_poly)
ridge_r2_poly = r2_score(y_test, y_pred_ridge_poly)

print(log_report_poly)
print("Ridge MSE:", ridge_mse_poly)
print("Ridge R² Score:", ridge_r2_poly)



# In[20]:


# Re-importing necessary libraries and reloading the dataset due to the reset of the code execution environment
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the dataset
dataset_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(dataset_path)

# Preparing the data for logistic and linear regression
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions for Logistic Regression
y_pred_log = log_reg.predict(X_test)

# Evaluating the Logistic Regression model
log_reg_accuracy = accuracy_score(y_test, y_pred_log)
log_reg_conf_matrix = confusion_matrix(y_test, y_pred_log)
log_reg_class_report = classification_report(y_test, y_pred_log)

# Linear Regression Model using 'avg_training_score' as independent variable
X_linear = data[['avg_training_score']]
y_linear = data['is_promoted']

# Splitting the dataset for linear regression
X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train_linear, y_train_linear)

# Predictions for Linear Regression
y_pred_linear = linear_reg.predict(X_test_linear)

# Evaluating the Linear Regression model
mse_linear = mean_squared_error(y_test_linear, y_pred_linear)
r2_linear = r2_score(y_test_linear, y_pred_linear)

# Visualization for Logistic Regression (Confusion Matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(log_reg_conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Visualization for Linear Regression (Scatter plot of actual vs predicted values)
plt.figure(figsize=(10, 6))
plt.scatter(X_test_linear, y_test_linear, color='blue', label='Actual')
plt.plot(X_test_linear, y_pred_linear, color='red', label='Predicted')
plt.title('Actual vs Predicted Promotions (Linear Regression)')
plt.xlabel('Average Training Score')
plt.ylabel('Promotion (0 or 1)')
plt.legend()
plt.show()

log_reg_accuracy, log_reg_class_report, mse_linear, r2_linear




# In[21]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/sairamoharana/Downloads/cleaned_data.csv'
data = pd.read_csv(file_path)

# Separate the features and the target variable
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train_scaled, y_train)

# Predict the test set results
y_pred = logreg.predict(X_test_scaled)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report
class_report = classification_report(y_test, y_pred)

# Display the accuracy and the classification report
accuracy, class_report


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the cleaned_data.csv is already loaded into the 'data' DataFrame
# X = data.drop('is_promoted', axis=1)
# y = data['is_promoted']
# For demonstration purposes, we'll proceed with the 'X' and 'y' as defined

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Evaluating the Logistic Regression model
log_reg_accuracy = accuracy_score(y_test, y_pred_log)
log_reg_class_report = classification_report(y_test, y_pred_log, output_dict=True)

# Displaying the Logistic Regression classification report in table format
print("\nLogistic Regression Classification Report:")
log_reg_report_df = pd.DataFrame(log_reg_class_report).transpose()
print(log_reg_report_df)

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train[['avg_training_score']], y_train)
y_pred_linear = linear_reg.predict(X_test[['avg_training_score']])

# Evaluating the Linear Regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Displaying Linear Regression metrics in table format
print("\nLinear Regression Metrics:")
linear_metrics_df = pd.DataFrame({'MSE': [mse_linear], 'R-squared': [r2_linear]})
print(linear_metrics_df)

# Confusion Matrix for Logistic Regression
conf_matrix = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Scatter Plot for Linear Regression
plt.figure(figsize=(10, 8))
plt.scatter(X_test['avg_training_score'], y_test, color='blue', label='Actual')
plt.plot(X_test['avg_training_score'], y_pred_linear, color='red', label='Predicted')
plt.title('Actual vs Predicted Promotions (Linear Regression)')
plt.xlabel('Average Training Score')
plt.ylabel('Promotion (0 or 1)')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




