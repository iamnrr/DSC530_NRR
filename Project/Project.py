
# coding: utf-8

# # Exploratory Data Analysis
# 

# Author: Raghu Raman Nanduri
# 
# Date: March 03,2019
# 
# Course: DSC530 - Term Project
# 
# Source Link: https://www.kaggle.com/noriuk/us-education-datasets-unification-project/version/4
# 
# Data set: finance_districts.csv
#     

# Statistical/Hypothetical Question:
#     By exploring this data set regarding financial school districts and their enrollment numbers, I want to find out whether there is a any statistical correlation between total revenues of the state and number of enrollment numbers for the school districts. If there is correlation, how much have an impact Total Revenues have on enrollment.
# 

# # Import libraries and data set

# In[11]:



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import scipy.stats as stats
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('bmh')


# In[12]:


# Importing data set
findistdf = pd.read_csv('src/finance_districts.csv')
findistdf.head()


# # Variables

# In[13]:


findistdf.columns


# # Description of the data set
The data set consists of financials of each school district in each state for different years. It has the following variables:

STATE - State of Financial School district
ENROLL - The U.S. Census Bureau's count for students in the state. Should be comparable to GRADES_ALL
NAME - Name of the school district
YRDATA  - Year that the record pertains to
TOTALREV: The total amount of revenue for the state. 
            TFEDREV - Federal Revenue
            TSTREV  - State Revenue
            TLOCREV  - Local Revenue
TOTALEXP: The total expenditure for the state.
            TCURINST - Instruction Expenditure
            TCURSSVC - Supportive Services Expenditure
            TCURONON - Other Expenditure
            TCAPOUT - Capital Outlay Expenditure
            
Note: link for data set (finance_districts.csv)- https://www.kaggle.com/adrian1acoran/starter-u-s-education-datasets-4a0c2b4b-7/data
# # Data Cleansing

# In[14]:


# Let us ignore the variables that are not part of this analysis
findistdf_orig = findistdf # taking backup of the original dataset

findistdf  = findistdf[['STATE', 'ENROLL', 'NAME', 'YRDATA', 'TOTALREV', 'TFEDREV', 'TSTREV', 'TLOCREV', 'TOTALEXP']]


# In[15]:


# Lets look at the basic information of the data set
findistdf.info()


# Looks like ENROLL variable data is not available for the rows. For this analysis, discard the rows that have null or nan values in the 2 important variables - ENROLL and TOTALREV
# 

# In[16]:


findistdf = findistdf.dropna(subset = ['ENROLL',  'TOTALREV'])
findistdf_orig2 = findistdf
findistdf.info()


# # Distributions - Histograms, Outliers

# In[17]:


# Getting basic stats
findistdf.describe()


# Drawing Histograms and Density plots

# In[18]:



plt.figure(figsize=(9, 8))
#sns.distplot(findistdf['ENROLL'], color='b', bins=25, hist_kws={'alpha': 0.4});
sns.distplot(findistdf['ENROLL'], color='b',hist = True,  bins=25) #, hist_kws={'alpha': 0.4});


# In[19]:


findistdf['ENROLL'].hist(bins = 80)
#plt.locator_params(nbins=20)
plt.xlabel('Enroll')
plt.ylabel('Frequency')
plt.title('Histogram of ENROLL')


# In[20]:


plt.figure(figsize=(9, 8))
#sns.distplot(findistdf['ENROLL'], color='b', bins=25, hist_kws={'alpha': 0.4});
sns.distplot(findistdf['TOTALREV'], color='b',hist = True,  bins=25) #, hist_kws={'alpha': 0.4});


# In[21]:


# Trying to find apt number of bins from min to max
findistdf['TOTALREV'].hist(bins = 275,  figsize=[12,6])
plt.locator_params(nbins=20)
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.title('Histogram of TOTAL REVENUE')


# Both ENROLL and TOTALREV variables are asymmetrically (positively) skewed with longer tail towards the higher values.

# Clearly there are some outliers in ENROLL and TOTALREV that are skewing the distributions heavily. 

# In[22]:


findistdf.groupby('STATE').TOTALREV.agg(['min', 'max', 'mean', 'var'])


# In[23]:


findist_bystate = pd.DataFrame()
findist_bystate = findistdf.groupby('STATE', as_index = False)['TOTALREV', 'ENROLL'].max()


# In[24]:


#Top 5 states with most total revenue
findist_bystate.sort_values('TOTALREV',ascending=False)[0:5]


# In[25]:


#Top 5 states with most total revenue
findist_bystate.sort_values('ENROLL',ascending=False)[0:5]


# Clearly California and NewYork are in a different league with respective to Total Revenue and Enroll numbers. So, let's see how the distribution will be if we separate these states from the data set

# In[26]:


findistdf_orig3 = findistdf # taking backup copy of df

# separating the records of 'New York' and 'California' from the data set
findistdf = findistdf[(findistdf.STATE != 'NEW_YORK') & (findistdf.STATE != 'CALIFORNIA' )]

# removing recrods with 0 revenue's and enrollments

findistdf = findistdf[(findistdf.ENROLL >0) & (findistdf.TOTALREV >0) &  (findistdf.TFEDREV >0) 
                      &  (findistdf.TSTREV >0) &  (findistdf.TLOCREV >0) &  (findistdf.TOTALEXP >0)  ]

len(findistdf)


# Plotting Density plots, histograms

# In[27]:


# Density plot for TOTALREV
plt.figure(figsize=(9, 8))
sns.distplot(findistdf['TOTALREV'], color='b', bins=25, hist_kws={'alpha': 0.4});


# In[28]:


# Trying to find apt number of bins from min to max
noofbins = np.arange(start=findistdf['TOTALREV'].min(), stop=findistdf['TOTALREV'].max(), step=50000)
print(len(noofbins))


# In[29]:



findistdf['TOTALREV'].hist(bins = 25,  figsize=[12,6])
plt.locator_params(nbins=20)
plt.xlabel('Total Revenue')
plt.ylabel('Frequency')
plt.title('Histogram of TOTAL REVENUE')


# In[30]:


# Density plot for ENROLL
plt.figure(figsize=(9, 8))
sns.distplot(findistdf['ENROLL'], color='b', bins=25, hist_kws={'alpha': 0.4});


# In[31]:


# Trying to find apt number of bins from min to max
noofbins = np.arange(start=findistdf['ENROLL'].min(), stop=findistdf['ENROLL'].max(), step=10000)
print(len(noofbins))


# In[32]:


findistdf['ENROLL'].hist(bins = 25,  figsize=[12,6])
#plt.locator_params(nbins=20)
plt.xlabel('Enroll')
plt.ylabel('Frequency')
plt.title('Histogram of ENROLL')


# Still most of the numbers are packed at the lower end of the scale for the TOTALREV and ENROLL

# # Plotting Histogram for the 5 variables

# In[33]:


colnames = ['ENROLL', 'TOTALREV', 'TFEDREV', 'TSTREV','TLOCREV', 'TOTALEXP']
plt.figure(figsize=(15,15))
for i in range(len(colnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    #print(colnames[i])
    plt.hist(findistdf[colnames[i]].dropna())
    plt.ylabel('Frequency')
    plt.xlabel(findistdf[colnames[i]].name)


# # Plotting Desnity plots for the 5 variables

# In[34]:


colnames = ['ENROLL', 'TOTALREV', 'TFEDREV', 'TSTREV','TLOCREV', 'TOTALEXP']
plt.figure(figsize=(15,15))
for i in range(len(colnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    #print(colnames[i])
    sns.distplot(findistdf[colnames[i]].dropna())


# # Plotting outliers using boxplot

# In[35]:


plt.figure(figsize=(15,15))
for i in range(len(colnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    #print(colnames[i])
    sns.boxplot(y=findistdf[colnames[i]].dropna())
    #column=findistdf[colnames[i]]
    #column.to_frame().boxplot(figsize=[4,8])
    #findistdf.boxplot(column=colnames[i], figsize=[4,8]);


# Clearly the distribution of the metrics is not normal and they are absolutely skewed towards with the lower end of the scale with long tail on right end of the scale. 
# 
# Let's transform the metrics into log form and see, how their distributions and histograms look like

# In[36]:


# Adding log transformed columns to the dataframe
findistdf['lg_ENROLL'] = findistdf['ENROLL'].apply(np.log)
findistdf['lg_TOTALREV'] = findistdf['TOTALREV'].apply(np.log)
findistdf['lg_TFEDREV'] = findistdf['TFEDREV'].apply(np.log)
findistdf['lg_TSTREV'] = findistdf['TSTREV'].apply(np.log)
findistdf['lg_TLOCREV'] = findistdf['TLOCREV'].apply(np.log)
findistdf['lg_TOTALEXP'] = findistdf['TOTALEXP'].apply(np.log)
findistdf.head()


# # Plotting histograms of log transformed columns

# In[37]:


lgcolnames = ['lg_ENROLL', 'lg_TOTALREV', 'lg_TFEDREV', 'lg_TSTREV','lg_TLOCREV', 'lg_TOTALEXP']
plt.figure(figsize=(15,15))
for i in range(len(lgcolnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    #print(colnames[i])
    plt.hist(findistdf[lgcolnames[i]].astype('float'))
    plt.title(findistdf[lgcolnames[i]].name)


# Histograms of log transformed variables doesn’t appear as much skewed as they were earlier without log transformations. All variables except for lg_TOTALEXP appear in unimodal distribution where as lg_TOTALEXP is in bimodal distribution.

# In[38]:


plt.figure(figsize=(15,15))
for i in range(len(lgcolnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    #print(colnames[i])
    sns.distplot(findistdf[lgcolnames[i]].dropna().astype('float'))
    plt.title(findistdf[lgcolnames[i]].name)


# As per the above density plots, all the vairable appear close to log normal distribution

# In[39]:


plt.figure(figsize=(15,15))
for i in range(len(lgcolnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    column=findistdf[lgcolnames[i]]
    column.to_frame().boxplot(figsize=[4,8])



# Log transformation gives a better representation of these variables with lesser outliers

# In[40]:


#Plotting histogram and KDE for total revenue
plt.figure(figsize=(30, 30))
for i in range(len(lgcolnames)): 
          
    if i == 0:   
        plt.subplot(3,2,1)
    if i == 1:   
        plt.subplot(3,2,2)
    if i == 2:   
        plt.subplot(3,2,3)
    if i ==3:   
        plt.subplot(3,2,4)
    if i == 4:   
        plt.subplot(3,2,5)
    if i == 5:  
        plt.subplot(3,2,6)

    axtr = findistdf[lgcolnames[i]].astype('float').hist(bins = 26, color = 'lightblue') #, normed=True)
    findistdf[lgcolnames[i]].plot(kind='kde', color='Green', ax=axtr,  figsize=[16,8])
    #plt.locator_params(nbins=20)
    #plt.title('Histogram - KDE for %s with mean(red), median(yellow) and mode(brown), %findistdf[lgcolnames[i]].name')
    plt.xlabel(findistdf[lgcolnames[i]].name);
    plt.axvline(findistdf[lgcolnames[i]].mean(),color='red',label='Mean')
    plt.axvline(findistdf[lgcolnames[i]].median(),color='yellow',label='Median')
    plt.axvline(findistdf[lgcolnames[i]].mode()[0],color='brown',label='Mode')
    plt.legend()
#plt.title('Histogram - KDE with mean(red), median(yellow) and mode(brown)')


# # Exploring data set by STATE and YEAR

# In[41]:


findist_bystateyr = pd.DataFrame()
findist_bystateyr = findistdf.groupby(['YRDATA', 'STATE']).sum()


# In[42]:


# Displaying top 5 total revenues by state and year
findist_bystateyr.sort_values('TOTALREV',ascending=False)[0:5]


# In[43]:


# Displaying bottom 5 total revenues by state and year
findist_bystateyr.sort_values('TOTALREV')[0:5]


# # Calculating Mean, Mode, Spread, and Tails 

# In[44]:


# statistics for whole data set(including California & New York - school districts)
findistdf_orig2.describe()


# In[45]:


def descchar(var): 
    """ 
    Function to print descriptive charecteristics of the variable
        Input: dataframe.columnname
        Returns: descriptive charecteristics like - mean, median, mode, spread, interquartile range, skew
        """
    
    print('  Mean, Median, Mode of %s, %f %f %f ' %(var.name, var.mean(), var.median(), var.mode()[0])) 
    print('  Spread - Variance, Standard deviation of %s, %f %f ' %(var.name, var.var(), var.std())) 
    print('  Skew of %s, %f ' %(var.name, var.skew())) 
    print('  Interquartile range of %s, %f %f %f' %(var.name, var.quantile(0.25), var.quantile(0.5), var.quantile(0.75))) 
    
    


# In[46]:


# statistics for entire data set with  California & New York school districts
for i in range(len(colnames)):
    print("Descriptive Characteristics for %s" % findistdf_orig2[colnames[i]].name)
    descchar(findistdf_orig2[colnames[i]])


# In[47]:


# statistics for data set without California & New York school districts
findistdf.describe()


# In[48]:


# statistics for data set without California & New York school districts
for i in range(len(colnames)):
    print("Descriptive Characteristics for %s" % findistdf[colnames[i]].name)
    descchar(findistdf[colnames[i]])


# Skew is far greater than 1, highlighting that the numbers for every column are skewed heavily towards right with long tail towards higher scale

# # Plotting PMFS

# In[49]:


#Top 5 states with most total revenue
findist_bystate.sort_values('TOTALREV',ascending=False)[0:5]


# In[50]:


# Bottom 5 states with most total revenue
findist_bystate.sort_values('TOTALREV',ascending=True)[0:5]


# In[51]:


# Comparing 2 states -FLORIDA and VERMONT 
flfindistdf =  findistdf[findistdf.STATE == 'FLORIDA']
vfindistdf =  findistdf[findistdf.STATE == 'VERMONT']


# In[52]:


import thinkstats2
import thinkplot


flfindistdfpmf = thinkstats2.Pmf(flfindistdf['TOTALREV'], label='FLORIDA')
vfindistdfpmf = thinkstats2.Pmf(vfindistdf['TOTALREV'], label='VERMONT')


# In[53]:


width=200000
axis = [0, 800, 0, 0.0005]

thinkplot.PrePlot(2, cols =2)
thinkplot.Hist(flfindistdfpmf, align = 'right', width = width)
thinkplot.Hist(vfindistdfpmf, align = 'left', width = width)
thinkplot.Config(xlabel = 'Total Revenue', ylabel = 'PMF')


# In[54]:


thinkplot.Pmf(flfindistdfpmf)

thinkplot.Pmf(vfindistdfpmf)


# In[55]:


thinkplot.PrePlot(2)
thinkplot.subplot(2)
#axis = [0, 800, 0, 0.0005]
thinkplot.Pmfs([flfindistdfpmf,vfindistdfpmf ])
thinkplot.Show(xlabel = 'Total Revenue', ylabel = 'PMF')  


# # Lets plot PMF of log transformed columns

# In[56]:


findistdf.columns


# In[57]:


lgflfindistdfpmf = thinkstats2.Pmf(flfindistdf['lg_TOTALREV'], label='FLORIDA')
lgvfindistdfpmf = thinkstats2.Pmf(vfindistdf['lg_TOTALREV'], label='VERMONT')


# In[58]:


width= 0.5
#axis = [0, 800, 0, 0.0005]

thinkplot.PrePlot(2, cols =2)
thinkplot.Hist(lgflfindistdfpmf, align = 'right',color="blue",  width = width)
thinkplot.Hist(lgvfindistdfpmf, align = 'left',color="orange",  width = width)
thinkplot.Config(xlabel = 'Total Revenue (log)', ylabel = 'PMF')


# In[59]:


fltr4 = np.array(flfindistdf['lg_TOTALREV'].dropna())
vtr4 = np.array(vfindistdf['lg_TOTALREV'].dropna())

range_lb = int(min([np.min(fltr4), np.min(vtr4)]))
range_ub = int(max([np.max(fltr4), np.max(vtr4)]))

nbr_bins =  range_ub - range_lb

pmf_fltr4 = np.histogram(np.array(fltr4), 
                                bins=nbr_bins, range=(range_lb, range_ub))
pmf_vtr4 = np.histogram(np.array(vtr4), 
                                bins=nbr_bins, range=(range_lb, range_ub))

width = 0.001
plt.bar(np.arange(range_lb, range_ub), pmf_fltr4[0],  align = 'center', color="blue", label="FLORIDA")
plt.bar(np.arange(range_lb, range_ub) + width, pmf_vtr4[0], align = 'edge', color="orange", label="VERMONT")

plt.xlabel("Total Revenue (log)")
plt.ylabel("probability")
plt.legend(loc="best")


# Based on the comparisons of PMF's Vermont - school districts are more likely to have lesser total revenues than Illinois school districts

# # Calculating CDF

# In[60]:


# CDF for total revenue
cdttlrev = np.sort(findistdf.TOTALREV) #ENROLL)
hist = np.histogram(cdttlrev, bins=15, range=(0,400000))
sz = len(cdttlrev)

plt.step(hist[1][:-1], np.cumsum(hist[0])/sz)
plt.xlabel("Total Revenue")
plt.ylabel("CDF(x)")
plt.title("CDF of TOTAL Revenue")


# 95% of the total revenues for all school districts are less than 50,000

# In[61]:


# comparing CDF's of total revenue for school districts in different states

vfindistdf =  findistdf[findistdf.STATE == 'VERMONT']
ilfindistdf =  findistdf[findistdf.STATE == 'ILLINOIS']

actvtr = np.array(vfindistdf['TOTALREV'].dropna())
actiltr = np.array(ilfindistdf['TOTALREV'].dropna())


# In[62]:


# CDF for Vermont state
vcdttlrev = np.sort(actvtr) #ENROLL)
histv = np.histogram(vcdttlrev, bins=20, range=(0,500000))
szv = len(vcdttlrev)

# CDF for Illionis state
ilcdttlrev = np.sort(actiltr) #ENROLL)
histil = np.histogram(ilcdttlrev, bins=20, range=(0,500000))
szil = len(ilcdttlrev)


plt.step(histv[1][:-1], np.cumsum(histv[0])/szv, color = 'blue', label = 'Total Revenue - Vermont')
plt.step(histil[1][:-1], np.cumsum(histil[0])/szil, color = 'red', label = 'Total Revenue - Illinois')
plt.xlabel("Total Revenue")
plt.ylabel("CDF(x)")
plt.title('CDF of Vermont Vs Illinois')
plt.legend()


# Overall school districts in Illinois have higher total revenue than Vermont and 98% of the total revenues for all school districts in Illinois is less than 100,000. Whereas for Vermont, almost 97% of the total revenues for school districts are below 50000$.
# Put it in another way, Illinois school districts have higher chance of having more Total Revenue.

# # Plotting analytical distributions

# In[63]:


# PLotting analytical distributions for total revenue

# calculate the mean and standard deviation

mean_tr = np.mean(findistdf.TOTALREV)
mean_tr

std_tr = np.std(findistdf.TOTALREV)
std_tr

# plot a normal distribution with the mean and standard deviation of total revenue

low = min(findistdf.TOTALREV)
high = max(findistdf.TOTALREV)


xs = np.linspace(low, high, 100)
ps = stats.norm.cdf(xs, mean_tr, std_tr)

plt.plot(xs, ps, label='model', color='0.6')

plt.title('Total Revenue vs CDF for normal distribution')
plt.xlabel('Sample of normal distribution')
plt.ylabel('CDF')
plt.legend()


# In[64]:


# CDF for total revenue
cdttlrev = np.sort(findistdf.TOTALREV) #ENROLL)
hist = np.histogram(cdttlrev, bins=15, range=(0,max(findistdf.TOTALREV)))
sz = len(cdttlrev)


plt.plot(hist[1][:-1], np.cumsum(hist[0])/sz, color = 'b', label = 'smooth')

plt.step(hist[1][:-1], np.cumsum(hist[0])/sz, color = 'r', label = 'step')

plt.title('Total Revenue vs CDF for normal distribution')
plt.xlabel('Total Revenue')
plt.ylabel('CDF')
plt.legend()


# In[65]:


# Overlapping distribution plots of normal distribution and cdf distribution of total revenues

plt.plot(xs, ps, label='model', color='0.6')
plt.plot(hist[1][:-1], np.cumsum(hist[0])/sz, label='data')

plt.title('CDF of Total Revenue vs CDF of normal distribution')
plt.xlabel('Sample of normal distribution')
plt.ylabel('CDF')

plt.legend()


# From the above overlapping plot of normal distribution vs total revenue, we can infer that Normal distribution doesn’t represent the total revenue. Let us try to see if log normal distribution is applicable.  
# To find out I will use the log transformed total revenue column - findistdf.lg_TOTALREV

# In[66]:


mean_ltr = np.mean(findistdf.lg_TOTALREV)
mean_ltr

std_ltr = np.std(findistdf.lg_TOTALREV)
std_ltr

# plot a normal distribution with the mean and standard deviation of total revenue

llow = min(findistdf.lg_TOTALREV)
lhigh = max(findistdf.lg_TOTALREV)

 
lxs = np.linspace(llow, lhigh, 10000)
lps = stats.norm.cdf(lxs, mean_ltr, std_ltr)

plt.plot(lxs,lps, label='model', color='0.6')


# In[67]:



mean_ltr = np.mean(findistdf.lg_TOTALREV)
mean_ltr

std_ltr = np.std(findistdf.lg_TOTALREV)
std_ltr

# plot a normal distribution with the mean and standard deviation of total revenue

llow2 = mean_ltr - 4 * std_ltr 
lhigh2 = mean_ltr + 4 * std_ltr 

 
lxs2 = np.linspace(llow2, lhigh2, 10000)
lps2 = stats.norm.cdf(lxs2, mean_ltr, std_ltr)

plt.plot(lxs2,lps2, label='model', color='0.6')
plt.title('Log (Total Revenue) vs CDF of log normal distribution')
plt.xlabel('Sample of log normal distribution')
plt.ylabel('CDF')


# In[68]:


# CDF for log transformed total revenue
cdttlrev4 = np.sort(findistdf.lg_TOTALREV) #ENROLL)
hist = np.histogram(cdttlrev4, bins=1000, range=(0,max(findistdf.lg_TOTALREV)))
sz = len(cdttlrev4)


plt.plot(hist[1][:-1], np.cumsum(hist[0])/sz)

plt.title('Log (Total Revenue) vs CDF')
plt.xlabel('Log (Total Revenue)')
plt.ylabel('CDF')

#plt.legend()


# In[69]:


# Overlapping distribution plots of log normal distribution and cdf distribution of log (total revenues)

plt.plot(lxs,lps, label='model', color='0.6')
plt.plot(hist[1][:-1], np.cumsum(hist[0])/sz, label='data')

plt.title('CDF of Log(Total Revenue) vs CDF of log normal distribution')
plt.xlabel('Log(Total Revenue)')
plt.ylabel('CDF')
plt.legend()


# From the above overlapping plot of log normal distribution vs log(total revenue), we can infer that log normal distribution perfectly fits for the variable total revenue

# # Probability plots for total revenue and Log (total revenue)

# In[70]:


xs = [-5, 5]
# y(x) = mean + std * x, here mean and standard deviation are from Total Revenue
ys = mean_tr + std_tr * np.sort(xs) 
plt.plot(xs, ys, color='red', label='model')

plt.xlabel('Z')
plt.ylabel('Total Revenue (linear scale)')


# In[71]:


n = len(findistdf.TOTALREV)
xs2 = np.sort(np.random.normal(0, 1, n))
ys2 = np.sort(np.array(findistdf.TOTALREV))

plt.plot(xs2, ys2, color='blue', label='data')
plt.plot(xs, ys, color='red', label='model')

plt.title('Normal Probability plot on linear scale')
plt.xlabel("Z - standard deviation")
plt.ylabel('Total Revenue')
plt.legend()


# In[72]:


# Normal probability plot of log normal form

xs = [-5, 5]
# y(x) = mean + std * x, here mean and standard deviation are from log transformed Total Revenue
ys = mean_ltr + std_ltr * np.sort(xs)
plt.plot(xs, ys, color='red', label='model')


# In[73]:


n = len(findistdf.lg_TOTALREV)
xs2 = np.sort(np.random.normal(0, 1, n))
ys2 = np.sort(np.array(findistdf.lg_TOTALREV))

plt.plot(xs2, ys2, color='blue', label='data')
plt.plot(xs, ys, color='red', label='model')

plt.title('Normal Probability plot on log scale')
plt.xlabel("Z - standard deviation")
plt.ylabel("Log (Total Revenue)")

plt.legend()


# From the above two normal probability plots, we can infer that data deviates substantially from normal model 
# where as log normal model fits perfectly to the data with in 2 standard deviations (between -2 to 2) but deviates from the log normal model significantly for the school districts with lower and higher end of the (log) total revenue scale.

# # Scatter plots and Correlation analysis

# In[74]:


findistdf.head()


# In[75]:


#plt.style.use('ggplot')
plt.style.use('seaborn')
findistdf.plot(x= 'TOTALREV', y = 'ENROLL', kind = 'scatter' ) 
plt.title('Scatter plot of Total Revenue Vs Enroll')
plt.show()


# The above chart shows that school districts with higher total revenue has better enrollment than the school districts with lower total revenue, agrees with one of our assumptions.

# In[76]:


# Scatter plot with jitter
jitter = 20000
TOTALREV = findistdf.TOTALREV + np.random.uniform(-jitter, jitter)
plt.scatter(TOTALREV, findistdf.ENROLL, alpha = 0.3, s = 10)
plt.title('Scatter plot of Total Revenue Vs Enroll')
plt.show()


# In[77]:


# Scatter plot between log(total revenue) and enroll

jitter = 1
lg_TOTALREV = findistdf.lg_TOTALREV + np.random.uniform(-jitter, jitter)
plt.scatter(lg_TOTALREV, findistdf.ENROLL, alpha = 0.3, s = 10)
plt.title('Scatter plot of log(Total Revenue) vs Enroll')
plt.show()


# In[78]:


# Scatter plot between log(total revenue) and log(enroll)

jitter = 0.3
ENROLL4 =  findistdf.lg_ENROLL + np.random.uniform(-jitter, jitter)
TOTALREV4 = findistdf.lg_TOTALREV + np.random.uniform(-jitter, jitter)
plt.scatter(TOTALREV4, ENROLL4, alpha = .2, s = 10)
plt.title('Scatter plot of log(Total Revenue) vs Enroll')
plt.show()


# # Characterizing Relationships

# In[79]:



findistdf2 = findistdf.dropna(subset = ['ENROLL',  'TOTALREV'])
bins = np.arange(0, 30000000, 200000)
indicies = np.digitize(findistdf2.TOTALREV, bins)

grps = findistdf2.groupby(indicies)


# In[80]:


for i, group in grps:
    print(i, len(group))


# In[81]:


mean_tr = [group.TOTALREV.mean() for i, group in grps]
cdfs = [thinkstats2.Cdf(group.ENROLL) for i, group in grps]


# In[82]:


for percent in [75, 50, 25]:
    enroll_percentiles = [cdf.Percentile(percent) for cdf in cdfs]
    label = '%dth' % percent
    thinkplot.Plot(mean_tr, enroll_percentiles, label=label)
    thinkplot.Config(xlabel='Total Revenue',  ylabel='ENroll',legend=True)
    plt.title('Percentile plots of Total Revenue Vs Enroll')


# Above percentiles plotof of Total Revenue Vs Enroll, relationship is linear upto 3500000$, after that relationship is going in the wrong direction.

# In[83]:


descchar(findistdf['TOTALREV'])


# # Covariance and Correlation

# In[84]:


# Covariance

def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov

Cov(findistdf2.TOTALREV, findistdf2.ENROLL)


# In[85]:


# Calculating Correlation between total revenue and Enroll in school districts
def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr

Corr(findistdf2.TOTALREV, findistdf2.ENROLL)


# Correlation value of 0.95 indicates that total revenue and enroll variables are strongly and positively correlated; and it implies that school districts with higher total revenue tend to have higher enrollments in 
# those schools. 
# But our distributions are highly skewed and are not normal distributions, so let’s find out the Spearman's Rank correlation as well.

# In[86]:


def SpearmanCorr(xs, ys):
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return Corr(xranks, yranks)

SpearmanCorr(findistdf2.TOTALREV, findistdf2.ENROLL)


# # Hypothesis Testing

# Defining Null Hypothesis: My earlier assumption is that school districts with higher total revenue will have higher enrollments in the school. Based on that, my Null hypothesis is that there is no relationship between Total revenue and school enrollments for school districts. The p-value for this correlation testing is to find out the probability of having such a high observed correlation of 0.95 by pure chance should be significant (pvalue > 0.05).
# 
# Let us find out with Hypothesis testing.

# In[87]:


findistdf.head()


# In[88]:


findist_bystate_mn = pd.DataFrame()
findist_bystate_mn = findistdf.groupby('STATE', as_index = False)['TOTALREV', 'ENROLL'].mean()
findist_bystate_mn.sort_values('TOTALREV',ascending=False)[0:5]


# In[89]:


findist_bystate_mn.sort_values('ENROLL',ascending=False)[0:5]


# In[90]:


Hyp_df = findistdf.dropna(subset = ['ENROLL',  'TOTALREV'])
Hyp_df.info()


# In[91]:


# Pearson correlation
pecorr, p = stats.pearsonr(Hyp_df.TOTALREV, Hyp_df.ENROLL)
pecorr, p


# In[92]:


# Spearman correlation
corr2, p2 = stats.spearmanr(Hyp_df.TOTALREV, Hyp_df.ENROLL)
corr2, p2


# In[93]:


trenr_df = Hyp_df[['TOTALREV', 'ENROLL']]


# In[94]:


stat, p, dof, expected = stats.chi2_contingency(trenr_df)
stat, p, dof, expected


# Method for testing this null hypothesis is to randomly generate values for total revenue and enroll with the same mean and standard deviation of the current data set and calculate the Correlation and P- value for that sample data set. Repeat the process for some iterations (100)
# 

# In[95]:



def samplepermute(iters = 100):
    ''' 
    Function to permutate the TOTALREV randomly and calculate Correlation, p value and Covariance of that  data set.
    iters is number of iterations of test
    returns: 
    smpcorr - sample correlation
    smppval - sample pvalue
    count/iters - % of samples that have sampled correlation greater than observed correlation
    '''
    smpcorr = []
    smppval = []
    count = 0
    df = pd.DataFrame()
    for j in range(iters):


        corr, p = stats.pearsonr(np.random.permutation(Hyp_df.TOTALREV),Hyp_df.ENROLL) 
        cov  = Cov(np.random.permutation(Hyp_df.TOTALREV), Hyp_df.ENROLL)


        if abs(corr) >= pecorr:
            count += 1
        smpcorr.append(corr)
        smppval.append(p)
    return smpcorr, smppval, count/iters
        
    


# In[96]:


test2corr, test2p, test2count = samplepermute( iters = 100)
test2count


# The probability of having such a high (observed - 0.95) correlation between Total Revenue & Enroll by chance is 0. Hence null hypothesis that there is no correlation between Total Revenue and Enroll is false.

# # Regression Analysis

# In[97]:


findistdf1 = findistdf.dropna(subset = ['ENROLL',  'TOTALREV'])


# In[98]:


#PLotting scatter plot between Total Revenue and Enroll
#plt.style.use('ggplot')
plt.style.use('seaborn')
findistdf1.plot(x= 'TOTALREV', y = 'ENROLL', kind = 'scatter' )
plt.xlabel('Total Revenue')
plt.ylabel('Enroll')
plt.title('Scatter plot of Total Revenue Vs Enroll with Best fit line')
plt.show()


# # Linear Least Square Model

# In[99]:


# plotting scatter plot again
plt.style.use('seaborn')
findistdf1.plot(x= 'TOTALREV', y = 'ENROLL', kind = 'scatter' )
plt.xlabel('Total Revenue')
plt.ylabel('Enroll')
plt.title('Scatter plot of Total Revenue Vs Enroll')
plt.show()


# In[100]:


#Calculating slope & iter

meantr = np.mean(findistdf1.TOTALREV)
meanenr = np.mean(findistdf1.ENROLL)

slope = (meantr * meanenr - np.mean(findistdf1.TOTALREV * findistdf1.ENROLL))/ (meantr**2 -  np.mean(findistdf1.TOTALREV **2) )

inter = meanenr - meantr*slope

meantr, meanenr, slope, inter


# In[101]:


# Drawing best fit line

plt.scatter(findistdf1.TOTALREV, findistdf1.ENROLL)
TR_minmax = [np.min(findistdf1.TOTALREV), np.max(findistdf1.TOTALREV)]


Regressionline = [slope*t + inter for t in TR_minmax ]

plt.plot(TR_minmax, Regressionline, color = "red", alpha =0.3)
plt.xlabel('Total Revenue (Independant variable)')
plt.ylabel('Enroll (dependant variable)')
plt.title('Scatter plot of Total Revenue Vs Enroll with Linear line fit')


# Above plot confirms the linear relationship between Total Revenue and Enroll

# # Goodness of Linear Least Square fit

# Calculating Residuals, RMSE, Coeffecient of Determination

# Goodness of linear least square fit can be found by comparing the Root mean square error between with model and without model. 

# In[102]:


# PLotting residuals
def Residuals(xs, ys, inter, slope):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res

findistdf1['residual'] = Residuals(findistdf1.TOTALREV, findistdf1.ENROLL, inter, slope)
#len(findistdf1['residual'])


# In[103]:


# Function to calculate Root mean squared error
def RMSE(ys, pred_ys):
    yactual = np.array(ys)
    ypred = np.array(pred_ys)
    error = (yactual - ypred)**2
    errmean = np.mean(error)
    errsqrt = sqrt(errmean)
    return errsqrt

xs = np.array(findistdf1.TOTALREV)
# predicted value for enroll using linear least square
pred_ys = [inter + (slope * x) for x in np.array(findistdf1.TOTALREV)]
RMSE(findistdf1.ENROLL, pred_ys)


# In[104]:


# Calculating Root Mean Square Error (RMSE) - Standard deviation of residuals
np.std(findistdf1['residual']), np.std(findistdf1['ENROLL'])


# Without any model, RMSE of predicted Enroll numbers is represented by its standard deviation – which here in this case is 10435.
# 
# With Linear Least Square fit model, RMSE of predicted Enroll numbers from known Total Revenues is calculated by finding the residuals from prediction (Observed Enroll – Predicted Enroll)  and finding the standard deviation from the residual. In this case it is 3267.
# 
# As predicting Enrollment numbers with Linear Least Square model results in lesser standard deviation, in this case knowing the total revenue and predicting enrollment numbers from it has significantly helped for better prediction and reducing the error.
# 

# In[105]:


# Coeffecient of Determination:
resid_var = np.var(findistdf1['residual'])
enroll_var = np.var(findistdf1['ENROLL'])

CoeffD = 1 - resid_var/enroll_var
CoeffD


# CoeffD of 0.90 indicates that total revenue helps predict almost 90% of the variance in the enrollment numbers for school districts.

# # Plotting residuals

# In[106]:


bins = np.arange(min(findistdf1.TOTALREV), max(findistdf1.TOTALREV), 250000)
indices = np.digitize(findistdf1.TOTALREV, bins)
groups = findistdf1.groupby(indices)

trbin_means = [group.TOTALREV.mean() for _, group in groups][1:-1]
#len(trbin_means) = 50


# In[107]:


cdfs = [thinkstats2.Cdf(group.residual) for _, group in groups][1:-1]


# In[108]:


def PlotPercentiles(trbin_means, cdfs):
    thinkplot.PrePlot(3)
    for percent in [75, 50, 25]:
        weight_percentiles = [cdf.Percentile(percent) for cdf in cdfs]
        label = '%dth' % percent
        thinkplot.Plot(trbin_means, weight_percentiles, label=label)


# In[109]:


PlotPercentiles(trbin_means, cdfs)

thinkplot.Config(xlabel="Total Revenue", ylabel='Residuals')


# The residual plots are not straight lines, indicates that relationship between total revenue and enroll is non - linear. 
# The gap between inter quartile residuals is most at the total revenue of 3 Million.
# 

# In[110]:


# Plotting best fit line with stats modules
plt.plot(findistdf1.TOTALREV, findistdf1.ENROLL, 'o', label='original data')
plt.plot(findistdf1.TOTALREV, inter + slope*findistdf1.TOTALREV, 'r', label='fitted line')
plt.legend()
plt.show()


# # Testing Linear Model

# In[111]:


#To estimate the sampling distribution of inter and slope, I'll use resampling.

def SampleRows(df, nrows, replace=False):
    """Choose a sample of rows from a DataFrame.

    df: DataFrame
    nrows: number of rows
    replace: whether to sample with replacement

    returns: DataDf
    """
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample

def ResampleRows(df):
    """Resamples rows from a DataFrame.

    df: DataFrame

    returns: DataFrame
    """
    return SampleRows(df, len(df), replace=True)


# In[112]:


def SamplingDistributions(findistdf1, iters=101):
    inters = []
    slopes = []
    for _ in range(iters):
        sample = ResampleRows(findistdf1)
        TOTALREV = sample.TOTALREV
        ENROLL = sample.ENROLL
        
        slope = (meantr * meanenr - np.mean(findistdf1.TOTALREV * findistdf1.ENROLL))/ (meantr**2 -  np.mean(findistdf1.TOTALREV **2) )
        slopes.append(slope)
        inter = meanenr - meantr*slope
        inters.append(inter)

    return inters, slopes


# In[113]:


inters, slopes = SamplingDistributions(findistdf1, iters=100)


# In[114]:


slope_cdf = thinkstats2.Cdf(slopes)
pvalue = slope_cdf[0]
pvalue


# Probability that the slope in the sampling distribution falls below 0 (p-value) is 0; as it is less than 
# 0.001 indicating that the relation between Total Revenue and Enroll is statistically significant and not by chance.

# In[115]:


for slope, inter in zip(slopes, inters):
    fxs = np.sort(findistdf1.TOTALREV)
    fys =  inter + slope * fxs
    thinkplot.Plot(fxs, fys, color='gray', alpha=0.01)
    
thinkplot.Config(xlabel="Total Revenue", ylabel='Residuals', title = "Regression lines for various iterations")


# After repeated sampling the regression line roughly stayed in the same place, so it is a low variance model

# # Regression Analysis - Ordinary Least Square Model

# In[116]:


import statsmodels.formula.api as smf

formula = 'findistdf1.ENROLL ~ findistdf1.TOTALREV'
model = smf.ols(formula, data=findistdf1)
results = model.fit()
results.summary()


# In[117]:


inter = results.params['Intercept']
slope = results.params['findistdf1.TOTALREV']
inter, slope


# Interpreting the coefficients:
#     slope value of 0.095 infers that unit increase in total revenue is associated with 0.095 unit increase in enroll numbers for the school districts.

# In[118]:


slope_pvalue = results.pvalues['findistdf1.TOTALREV']
slope_pvalue


# As P-value is less than 0.001, the estimated slope is significant

# In[119]:


# Coefficient of determination
results.rsquared


# R- Square value of 0.90 shows that variation in enrollment can be explained by variation in Total Revenue upto 90%. As more variance is being explained by the model, once again proves the fit of the model.

# In[120]:


#plotting residuals

pred_val = results.fittedvalues.copy()
true_val = findistdf1['ENROLL'].values.copy()
residual = true_val - pred_val


# In[121]:


fig, ax = plt.subplots(figsize=(10,10))
residplot = ax.scatter(residual, pred_val)


# In[122]:


# Drawing normal probability plot

fig, ax = plt.subplots(figsize=(8,6))
_, (__, ___, r) = stats.probplot(residual, plot=ax, fit=True)

r**2
#stats.probplot(residual, plot=ax, fit=True)


# Above normality plot indicates that this model is good fit only between quartile -2 to +2. After that it 
# significantly deviates from linear model.

# In[123]:


# Confidence intervals
results.conf_int()


# In[124]:


# p-values for the model coefficients
results.pvalues


# Again p-values are way less than 0.05, indicating that the relation between dependent and independent variables is genuine.

# # Correlation matrix

# In[125]:


# Compute the correlation matrix
corr = findistdf1.corr()


# In[126]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,   cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # Multiple Linear Regression

# In[127]:


# understanding whether the enrollment in school districts depends on Total Revenue and STATE as well
formula2 = 'findistdf1.ENROLL ~ findistdf1.TOTALREV + findistdf1.STATE'
model2 = smf.ols(formula2, data=findistdf1)
results2 = model2.fit()
results2.summary()


# # Comparing model 1 vs model2:

# 1) Adjusted R-square of model2 is .909, better than Adjusted R-square of model1 (0.902) - indicating that model2 can explain slightly more variation in dependent variable compared to model1.
# 
# 2) AIC of model1 is 5.473e+06, which is slightly higher than AIC of model2 5.451e+06, indicating that model2 (enrollment as a function of total revenue and state) is slightly better model among the 2 models.
# 
# 3) HAWAII has the highest absolute coefficient - indicating that ENROLL'ment numbers change hugely with a single unit of variation in TOTALREV for that state. In other words, we can probably see more enrollment numbers for every same number of units increase in total revenue compared to all other states (all remaining things being constant).
# 
# 4) LOUISIANA has the lowest absolute coefficient value - indicating that ENROLL numbers will change at a slower rate compared to all other states for the same unit of increase in the total revenue (all remaining things being constant).
# 
# 5) Overall adding STATE to the ordinary least square model, improved the model very slightly but not significantly. But interesting aspect of adding STATE to the equation is it gives us insights into how each is the relationship between Total Revenue and Enroll for each STATE.
# 

# # Outcome of EDA:

# The assumption that I had before exploring this data set was that the school districts that are in higher revenue states will have more chance of higher enrollments in the school. After performing EDA, 
# I did find statistical correlation between Total Revenues and Enrollment of the school districts. So my assumption was correct. 
# One more observation that I made is not all states will respond similarly to the total revenue numbers. For example, take state like LOUISIANA, even though Total Revenues increase for this state, Enrollment numbers will not raise proportionally when compared to other states. 
# 

# # What was missing during the analysis?

# Having demographic information of each school districts like population, family size, number of school going kids, family income etc. would have added more sense to the analysis. Also one of these or couple of these could be the confounding factors that I highlighted above with STATE – LOUISIANA.

# # Variables that could have helped in the analysis?

# As states above, demographic information could have helped more in the analysis in finding the actual enrollment prediction for the school districts.

# # Assumptions made correct or incorrect?

# No. The assumptions that I made that enrollments in school districts are based on Total Revenues of the state was correct, backed by the higher correlation factor and the linear models

# # Challenges faced

# 
# In the selected data, there were some other aggregated data sets available, which I wanted to explore and compare with the financial school district data set that I selected. But some of the variables that I wanted to explore like GRADES_ALL_G etc. are not available through those sheets. I feel enrollment numbers depend a lot on population or number of families living in that school district. Having a demographic data for these school district would have been an interesting analysis, I would like to try. But that information was not available readily, so I couldn’t venture into that analysis. One more challenge is that some of the states like California, New York are too big to be compared with other smaller states like Vermont, but before realizing this when I performed EDA overall financial school district data set, numbers were highly skewed – and didn’t find an ideal number of bins to represent these variables into proper histogram. Plotting PDF and CDF without using the thinkstats2, thinkplot modules has been a challenge, at some places I gave up trying to figure out other means and ended up using these modules. I would like to explore these in free time. 
# 
