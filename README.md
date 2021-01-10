# Cereals Mini Project

## Contents
***
1. Assigning working directory
2. Importing the necessary libraries
3. Importing the data
4. Exploratory data analysis (EDA)
    * Basic commands
    * Exploring categorical variables
    * Exploring numerical variables
5. Imputing Missing Values
6. Feature Engineering
7. Relation between columns
8. Pair Plot
9. Conclusion

## Assigning working directory
---
```python
    import os
    os.chdir('path to your working directory')
```

## Importing the necessary libraries
---
```python
    # Libraries for data manupulation
    import pandas as pd
    import numpy as np

    # Libraries for visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # This is an optional argument only used for jupyter notebook users
    %matplotlib inline

    # Using a custom style for visualisation
    # To see all the custom styles available input: 'plt.style.available'
    plt.style.use('fivethirtyeight')
```

## Importing the data
---
Here we are using cereals dataset. You can download this dataset from [kaggle](https://www.kaggle.com/crawford/80-cereals)

To import the data we can use the following command
```python
cereals = pd.read_csv('path_name/cereals_data.csv')
```

## Exploratory Data Analysis (EDA)
---
In exploratory data analysis we are looking at the data and getting the feel of the structure and content of the data.

```python
    cereals.head()
```
```python
    	name	mfr	type	calories	protein	fat	sodium	fiber	carbo	sugars	potass	vitamins	shelf	weight	cups	rating
0	100%_Bran	N	C	70	4	1	130	10.0	5.0	6.0	280.0	25	3	1.0	0.33	68.402973
1	100%_Natural_Bran	Q	C	120	3	5	15	2.0	8.0	8.0	135.0	0	3	1.0	1.00	33.983679
2	All-Bran	K	C	70	4	1	260	9.0	7.0	5.0	320.0	25	3	1.0	0.33	59.425505
3	All-Bran_with_Extra_Fiber	K	C	50	4	0	140	14.0	8.0	0.0	330.0	25	3	1.0	0.50	93.704912
4	Almond_Delight	R	C	110	2	2	200	1.0	14.0	8.0	NaN	25	3	1.0	0.75	34.384843
```

This is the head of our data.

Through some further exploration and visualization we can identify some interesting patterns in cereals data.

### Basic Commands
---
Some basic commands include
1. shape
2. info()
3. head()
4. columns
5. unique()
6. describe()

By executing these commands on cereals dataframe we can understand the basic structure and content of the dataframe.

### Exploring Categorical Variables
---
Some of the analysis are given below:

* How many cereal products does each manufacturer(_mfr_) have?

![mfr_count](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/1.mfr_count.png?raw=true)

> We can see that 'K'(Kelloggs) and 'G'(General Mills) have the most cereal products

* How many counts are there in each _type_ of cereals?

![Type_count](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/2.type_count.png?raw=true)

> Most of the cereals are cold type cereals

* How many products are there on each _shelf_?

![shelf_count](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/3.shelf_count.png?raw=true)

> Most products are placed on shelf 3

* How many products of each manufacturer(_mfr_) are placed on each shelves(_shelf_)?

![mfr_shelf_grouping](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/4.grouped_mfr_shelf.png?raw=true)


### Exploring Numerical variables

Running this code will give you the Minimum, Maximum, Sum, Standard deviation and Kurtosis of all numerical columns
```python
# Selecting the numerical features
cols = ['calories', 'protein', 'fat', 'sodium', 'fiber',
       'carbo', 'sugars', 'potass', 'vitamins', 'weight', 'cups']

# Looping through the numerical columns and printing necessary statistics like min, max, sum etc...

# Here the enumerate will add an index(0,1,2,3,4....) to the cols list['calories','protein'.....], and we can unpack them
# using the variables 'num' and 'i'. 'num' wil hold the index values and 'i' will hold the list elements.

for num, i in enumerate(cols): 
    print(f'{num}. The Sample Statistics of {i}')
    print('Min: ', cereals[i].min(),'\n',
        'Max: ',cereals[i].max(),'\n',
        'Sum: ', cereals[i].sum(),'\n',
        'Skew: ', cereals[i].skew(),'\n',
        'Std Dev: ', cereals[i].std(),'\n',
        'Kurtosis: ', cereals[i].kurtosis())
    print('\n')
```

## Imputing Missing Values
---

This command will give you the sum of all missing values in each columns

> (Note: Missing values are not always in `np.nan` or `NaN` format, sometimes extreme values are used in place of missing values like '-1', '0','99', '-999' (or other combinations of 9s).)

```python
    cereals.isnull().sum()
```
Or you can use this command to visualize the missing data using a heatmap.

```python
    sns.heatmap(cereals.isnull(),cmap='viridis')
    plt.title('Null value heatmap')
```
![null_value_heatmap](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/5.null_heatmap.png?raw=true)

To fill the missing values we use the `df.fillna()` function.
```python
# .fillna() will fill NA/NaN values using the specified method.

# Here we are selecting only the carbo column from cereals to fill the Null values.

# cereals[cereals.mfr=='Q']['carbo'].mean() will give us the mean of 'carbo' where 'mfr' is 'Q'(Quaker Oats)

# inplace : bool, default--False
# If True, fill in-place. Note: this will modify any other views on this object 
#     (e.g., a no-copy slice for a column in a DataFrame).

cereals.carbo.fillna(cereals[cereals.mfr=='Q']['carbo'].mean(),inplace=True)
```

Similarly we can fill all the missing datas in each columns.

## Feature Engineering
---
In this section we will create a new column based on _rating_ column. This new column will be a categorical variable.

```python
# We are creating a new ratings column(rating_cat) by converting the ratings column to a categorical variable

# Bin values into discrete intervals.

# Use cut when you need to segment and sort data values into bins. This function is also useful for going from a
# continuous variable to a categorical variable. 

# For example, cut could convert ages to groups of age ranges. 

# Supports binning into an equal number of bins, or a pre-specified array of bins.


bins = [0,25,50,75,100]
names = ['below_avg','average','above_avg','high']

cereals['rating_cat'] = pd.cut(cereals['rating'], bins, labels=names)
```
## Relation Between Columns
---
The best tool to understand the relationship between the columns is correlation matrix.

We can construct a correlation heatmap for better understanding of the relationship between data.

This is the code to create a correlation matrix using a heatmap:
```python
plt.figure(figsize=(12,12))

# By passing corretalion matrix into a heatmap we can visualize the correlation between columns.
sns.heatmap(cereals.corr(),annot=True,fmt='.1g',vmin=-1,cmap='coolwarm',square=True)
plt.title('Correlation Matrix')

#_______annot_______
# For an even easier interpretation, an argument called annot=True should be passed as well, which helps display
# the correlation coefficient.

#_______fmt______
# There are times where correlation coefficients may be running towards 5 decimal digits. A good trick to reduce 
# the number displayed and improve readability is to pass the argument fmt =’.3g'or fmt = ‘.1g'

#_______vmin,vmax,center________
# There are times where the correlation matrix bar doesn’t start at zero, a negative number, or end at a
# particular number of choice—or even have a distinct center. All this can be customized by specifying these
# three arguments: vmin, which is the minimum value of the bar; vmax, which is the maximum value of the bar; and center
```

![correlation_heatmap](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/13.corr_heatmap.png?raw=true)

## Pair Plot
---
Another great tool to visualize the relation between numerical features is the pairplot from seaborn library.

The command to create a pair plot is fairly simple, but it has lots of information about the data.
```python
# Pair Plot : Plot pairwise relationships in a dataset.

# By default, this function will create a grid of Axes such that each numeric variable in data will by shared across 
# the y-axes across a single row and the x-axes across a single column. The diagonal plots are treated differently: 
#     a univariate distribution plot is drawn to show the marginal distribution of the data in each column.


with plt.style.context('bmh'):
    sns.pairplot(cereals.iloc[:,:7],kind='reg')
    
plt.title('Pair Plot 1')
# Here we are indexing first 6 columns(4 numerical columns).
```
![pair_plot](https://github.com/darshan-ds/cereals_mini_project/blob/master/plots/22.pair_plot1.png?raw=true)

Similarly we can construct pairplots for the remaining of the data.

> Here we are using the `kind='reg'`, so that we can see a regression line through the middle of the scatter plot. The diagonal of the pairplot is filled with histogram because we cannot do a scatter plot using the same column on x and y axis.

## Conclusion

From the cereals data we can get to the conclusion that some features like:
* protein
* fiber and
* potass
affects the rating of that cereal in a positive way, and some other features like:
* calories
* fat
* sodium
* sugars
affect the rating of the cereals in a negative way.
