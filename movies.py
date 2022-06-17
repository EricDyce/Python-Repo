import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from traitlets import default
plt.style.use("ggplot")
from matplotlib.pyplot import figure

matplotlib.rcParams["figure.figsize"] = (12,8) #adjusts the config of the plots we will create

#read in the data
df = pd.read_csv(r"C:\Users\Wojuola Daniel\Downloads\R2 Business Intel\movies.csv")

#take a sneak peek
#df.head()

#checking for missing data
for col in df.columns:
    missing_col = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, missing_col))

df.dtypes

#change datatype for budget

df['budget'] = df['budget'].astype('int64')
df['gross'] = df['gross'].astype('int64')

# correct the year column
df['correctyear'] = df['released'].astype('str').str[:4]

df.sort_values(by = ['released'], ascending= False)

# To display all rows
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_rows', 10)

# To Drop duplicates
df.drop_duplicates()
# To do it on a column
df['name'].drop_duplicates().sort_values(ascending= False)


# Plotting Budget against gross

plt.scatter(x= df['budget'], y= df['gross'])
plt.title("Budget VS Gross Earnings")
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')

plt.show()


sns.regplot(x='budget', y= 'gross', data= df)


# To check for outliers
df.boxplot(column = ['gross'])

# Checking degree of correlation
df.corr()
# df.corr(method='pearson')   other methods kendall spearman
# Shows only correlation for number columns and not strings

# Create visual correlation matrix
Correlation_matrix = df.corr()
sns.heatmap(Correlation_matrix, annot = True)

plt.title('Correlation Heatmap')
plt.show()

# Creating numeric values for string columns so we can access their correlation
df_numerized = df
for col in df_numerized.columns:
    if(df_numerized[col].dtype == 'object'):
        df_numerized[col] = df_numerized[col].astype('category')
        df_numerized[col] = df_numerized[col].cat.codes

print(df_numerized)

df_numerized.corr()


Correlation_matrix2 = df_numerized.corr()
sns.heatmap(Correlation_matrix2, annot = True)

plt.title('Fulldata Correlation Heatmap')
plt.show()


# To organize the corr matrix for better comparison using Unstacking
corr_matrix = df_numerized.corr()
corr_matrix_unstacked = corr_matrix.unstack()
print(corr_matrix_unstacked)

sorted_corr = corr_matrix_unstacked.sort_values()
highly_correlated = sorted_corr[(sorted_corr)> 0.6]
print(highly_correlated)