#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Liabraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


# Reading the DATA
dt=pd.read_csv(r"C:\Users\HP\Downloads\day.csv")


# In[3]:


# Data 
dt.head()


# In[4]:


# Column check
print(dt.columns)


# In[5]:


# understanding various other parameter of the data
dt.describe()


# In[6]:


# Spotting the NULL values
nv = dt.isnull().sum()
print(nv)


# In[7]:


# understanding various other parameter of the data
dt.describe()


# In[8]:


# Spotting the NULL values
nv = dt.isnull().sum()
print(nv)


# In[9]:


#Understanding the details column wise
dt.info()


# In[10]:


#Dropping 'instant' as it is not a significant variable
dt.drop(['instant'],axis=1,inplace=True)


# In[11]:


dt.head(10)


# In[12]:


#Date is not required as of now as year month weekday already provided hence we drop column 'dteday'
dt.drop(['dteday'],axis=1,inplace = True)


# In[13]:


dt.head(10)


# In[14]:


#Variable 'cnt' is the summation of 'casual' and 'registered' hence to make it more precise we will keep only 'cnt' which is also our target variable
dt.drop(['casual','registered'],axis=1,inplace=True)


# In[15]:


dt.head()


# In[16]:


#Finding out categorical variables
dt.info()


# In[17]:


#season yr mnth holiday weekday workingday weathersit are catergorical variables 
# Hence categorising the values of 'season' 'weekday' and 'weathersit' with respective integers
dt['season'].replace({1:"spring",2:"summer",3:"fall",4:"winter"},inplace=True)
dt['weekday'].replace({0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"},inplace=True)
dt['weathersit'].replace({1:"Clear clouded",2:"Misty",3:"Thunderstorm",4:"Heavy Rain and Fog"},inplace=True)
dt.head()


# In[18]:


#converting numeric values 
v=dt[['temp','atemp','hum','windspeed','cnt']]
v=v.apply(pd.to_numeric)


# In[19]:


dt.head(10)
dt.info()


# In[20]:


#Pair plots for Data Analysis
sns.pairplot(dt,vars=['temp','atemp','hum','windspeed',"cnt"])
plt.show()


# In[21]:


#Finding correlation using heatmap
plt.figure(figsize = (15,11))
sns.heatmap(dt.corr(),annot = True, cmap = "YlGnBu")
plt.show()


# In[22]:


#The correlation between atemp and temp is 0.99 from the above heatmap 
# Hence we can drop temp and consider only atemp instead
dt.drop(['temp'],axis=1,inplace=True) 
dt.head()


# In[23]:


#Categorical Data Visualisation using boxplot
plt.figure(figsize=(20,10))
plt.subplot(3,3,1)
sns.boxplot(x='season',y='cnt',data=dt)
plt.subplot(3,3,2)
sns.boxplot(x='yr',y='cnt',data=dt)
plt.subplot(3,3,3)
sns.boxplot(x='mnth',y='cnt',data=dt)
plt.subplot(3,3,4)
sns.boxplot(x='holiday',y='cnt',data=dt)
plt.subplot(3,3,5)
sns.boxplot(x='weekday',y='cnt',data=dt)
plt.subplot(3,3,6)
sns.boxplot(x='workingday',y='cnt',data=dt)
plt.subplot(3,3,7)
sns.boxplot(x='weathersit',y='cnt',data=dt)
plt.show()


# In[24]:


#From the boxplot we can find that The median of is almost constant for count of total rental bikes for all days of the week  i.e. aproximately 4500
#The count of rental bikes reduces in spring season and Thunderstorm weather condition and January month as well
#The count of rental bikes increases by a huge margin in 2019 and is more in the month of September and October 
# The count also increases when the climate is Clear clouded durng monsoon fall season


# In[25]:


dt.info()


# In[26]:


#CREATING DUMMY VARIABLES


# In[27]:


#Step 1 : Converting categorical variables to object
dt['mnth']=dt['mnth'].astype(object)
dt['season']=dt['season'].astype(object)
dt['weathersit']=dt['weathersit'].astype(object)
dt['weekday']=dt['weekday'].astype(object)
dt.info()


# In[28]:


#Step 2 : Creating variables for categorical data
seasons=pd.get_dummies(dt['season'],drop_first=True)
weekdays=pd.get_dummies(dt['weekday'],drop_first=True)
month=pd.get_dummies(dt['mnth'],drop_first=True)
weather=pd.get_dummies(dt['weathersit'],drop_first=True)


# In[29]:


dt=pd.concat([dt,seasons],axis=1)
dt=pd.concat([dt,weekdays],axis=1)
dt=pd.concat([dt,month],axis=1)
dt=pd.concat([dt,weather],axis=1)
dt.info()


# In[30]:


#We will have to delete the original columns and it will create confusion 
dt.drop(['season'],axis=1,inplace=True)
dt.drop(['weathersit'],axis=1,inplace=True)
dt.drop(['weekday'],axis=1,inplace=True)
dt.drop(['mnth'],axis=1,inplace=True)
dt.head()


# In[31]:


#Spit the data such as the train and test data are always in same row
from sklearn.model_selection import train_test_split

np.random.seed(0)
dt_train, dt_test = train_test_split(dt,train_size = 0.5, test_size = 0.2, random_state = 100)


# In[32]:


dt_train.head()


# In[33]:


dt_test.head()


# In[34]:


dt_train.columns


# In[35]:


#Scaling the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[36]:


num=['atemp','hum','windspeed','cnt']
dt_train[num]=scaler.fit_transform(dt_train[num])


# In[37]:


dt_train.head()


# In[38]:


dt_train.describe()


# In[39]:


y_train = dt_train.pop('cnt')
X_train = dt_train


# In[40]:


X_train.head()


# In[41]:


y_train.head()


# In[42]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[43]:


#Feature selection using RFE approach
#Step 1 : we will use mixed approach first using 15 variables

lm = LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm,15)
rfe=rfe.fit(X_train,y_train)


# In[44]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[45]:


colum = X_train.columns[rfe.support_]
colum


# In[46]:


X_train_rfe = X_train[colum]


# In[47]:


# STATSMODEL 
import statsmodels.api as sm
X_train_rfe2 = sm.add_constant(X_train_rfe)


# In[48]:


lm = sm.OLS(y_train,X_train_rfe2).fit()


# In[49]:


print (lm.summary())


# In[50]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[51]:


X_train_rfe2.head()


# In[52]:


# Dropping column workingday as the VIF is very high
X_train_rfe = X_train_rfe.drop(['workingday'],axis=1)


# In[53]:


import statsmodels.api as sm
X_train_rfe2 = sm.add_constant(X_train_rfe)


# In[55]:


lm1 = sm.OLS(y_train,X_train_rfe2).fit()


# In[57]:


print(lm1.summary())


# In[58]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[59]:


#Dropping column atemp as it has high VIF
X_train_rfe = X_train_rfe.drop(['atemp'],axis=1)


# In[60]:


X_train_rfe2 = sm.add_constant(X_train_rfe)
lmA = sm.OLS(y_train,X_train_rfe2).fit()
print(lmA.summary())


# In[61]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[62]:


#Dropping column hum as it has high VIF
X_train_rfe = X_train_rfe.drop(['hum'],axis=1)


# In[63]:


X_train_rfe2 = sm.add_constant(X_train_rfe)
lmB = sm.OLS(y_train,X_train_rfe2).fit()
print(lmB.summary())


# In[64]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[65]:


#Dropping column 'windspeed' as it has high VIF
X_train_rfe = X_train_rfe.drop(['windspeed'],axis=1)


# In[67]:


X_train_rfe3 = sm.add_constant(X_train_rfe)
lmC = sm.OLS(y_train,X_train_rfe3).fit()
print(lmC.summary())


# In[68]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[69]:


#All the above columns are having VIF < 2 hence not dropping column based on VIF


# In[70]:


X_train_rfe3 = sm.add_constant(X_train_rfe)
lmC = sm.OLS(y_train,X_train_rfe3).fit()
print(lmC.summary())


# In[71]:


#'Saturday' has a very high p value hence drop Saturday
X_train_rfe = X_train_rfe.drop(['Saturday'],axis=1)


# In[72]:


X_train_rfe4 = sm.add_constant(X_train_rfe)
lmD = sm.OLS(y_train,X_train_rfe4).fit()
print(lmD.summary())


# In[73]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[74]:


# Adding month 7 lets see
X_train_rfe[7]=X_train[7]
X_train_rfe.head()


# In[75]:


X_train_rfe5 = sm.add_constant(X_train_rfe)
lmE = sm.OLS(y_train,X_train_rfe5).fit()
print(lmE.summary())


# In[76]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[77]:


#let add month 3 and analyse
X_train_rfe[3]=X_train[3]
X_train_rfe.head()


# In[78]:


X_train_rfe6 = sm.add_constant(X_train_rfe)
lmF = sm.OLS(y_train,X_train_rfe6).fit()
print(lmF.summary())


# In[79]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[80]:


# 3 has a very high p value hence drop Saturday
X_train_rfe = X_train_rfe.drop([3],axis=1)


# In[81]:


X_train_rfe7 = sm.add_constant(X_train_rfe)
lmG = sm.OLS(y_train,X_train_rfe7).fit()
print(lmG.summary())


# In[82]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[83]:


#let add month 6 and analyse
X_train_rfe[6]=X_train[6]
X_train_rfe.head()


# In[84]:


X_train_rfe8 = sm.add_constant(X_train_rfe)
lmH = sm.OLS(y_train,X_train_rfe8).fit()
print(lmH.summary())


# In[85]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[86]:


# 'winter' has a very high p value hence drop winter
X_train_rfe = X_train_rfe.drop(['winter'],axis=1)


# In[87]:


X_train_rfe9 = sm.add_constant(X_train_rfe)
lmI = sm.OLS(y_train,X_train_rfe9).fit()
print(lmI.summary())


# In[88]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[89]:


# Lets add 6 and check the analysis
X_train_rfe[6]=X_train[6]
X_train_rfe.head()


# In[90]:


# The above step no 85 is repeated kindly ignore


# In[91]:


# Lets add 'Misty' and check the analysis
X_train_rfe['Misty']=X_train['Misty']
X_train_rfe.head()


# In[92]:


X_train_rfe10 = sm.add_constant(X_train_rfe)
lmJ = sm.OLS(y_train,X_train_rfe10).fit()
print(lmJ.summary())


# In[93]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[94]:


# Lets add 'Monday' and check the analysis
X_train_rfe['Monday']=X_train['Monday']
X_train_rfe.head()


# In[95]:


X_train_rfe11 = sm.add_constant(X_train_rfe)
lmK = sm.OLS(y_train,X_train_rfe11).fit()
print(lmK.summary())


# In[96]:


# 'Monday' has a very high p value hence drop winter
X_train_rfe = X_train_rfe.drop(['Monday'],axis=1)


# In[97]:


X_train_rfe11 = sm.add_constant(X_train_rfe)
lmK = sm.OLS(y_train,X_train_rfe11).fit()
print(lmK.summary())


# In[98]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[99]:


# Lets add 'Tuesday' and check the analysis
X_train_rfe['Tuesday']=X_train['Tuesday']
X_train_rfe.head()


# In[100]:


X_train_rfe12 = sm.add_constant(X_train_rfe)
lmK = sm.OLS(y_train,X_train_rfe12).fit()
print(lmK.summary())


# In[101]:


# 'Tuesday' has a very high p value hence drop it
X_train_rfe = X_train_rfe.drop(['Tuesday'],axis=1)


# In[102]:


X_train_rfe12 = sm.add_constant(X_train_rfe)
lmK = sm.OLS(y_train,X_train_rfe12).fit()
print(lmK.summary())


# In[103]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[104]:


# Lets add 'Wednesday' and check the analysis
X_train_rfe['Wednesday']=X_train['Wednesday']
X_train_rfe.head()


# In[105]:


X_train_rfe13 = sm.add_constant(X_train_rfe)
lmL = sm.OLS(y_train,X_train_rfe13).fit()
print(lmL.summary())


# In[106]:


# 'Wednesday' has a very high p value hence drop it
X_train_rfe = X_train_rfe.drop(['Wednesday'],axis=1)


# In[107]:


X_train_rfe13 = sm.add_constant(X_train_rfe)
lmL = sm.OLS(y_train,X_train_rfe13).fit()
print(lmL.summary())


# In[108]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[109]:


# Lets add 'Thursday' and check the analysis
X_train_rfe['Thursday']=X_train['Thursday']
X_train_rfe.head()


# In[110]:


X_train_rfe14 = sm.add_constant(X_train_rfe)
lmM = sm.OLS(y_train,X_train_rfe14).fit()
print(lmM.summary())


# In[111]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[112]:


# 'Thursday' has a very high p value hence drop it
X_train_rfe = X_train_rfe.drop(['Thursday'],axis=1)


# In[113]:


X_train_rfe14 = sm.add_constant(X_train_rfe)
lmM = sm.OLS(y_train,X_train_rfe14).fit()
print(lmM.summary())


# In[114]:


#Calculate VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
A = X_train_rfe
vif['Features'] = A.columns
vif['VIF'] = [variance_inflation_factor(A.values, i) for i in range(A.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# In[116]:


# WE HAVE CHECKED ALL THE COLUMNS . THE OUTCOME IS "lmJ" IS HAVING THE BEST RESULTS . SO WE WILL CONSIDER IT FOR THE FURTHER PROCESS


# In[117]:


#Predicting Values
y_train_cnt = lmJ.predict(X_train_rfe10)


# In[118]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[119]:


#Calculate RESIDUALS 
R = y_train - y_train_cnt


# In[120]:


# Plotting ERROR and check Normality
fig = plt.figure()
sns.distplot((R), bins = 25)
fig.suptitle('Error Indication', fontsize = 25)
plt.xlabel('Error', fontsize = 20)


# In[121]:


# Column verify
X_train_rfe10.columns


# In[122]:


print(X_train_rfe10.shape)
print(R.shape)


# In[123]:


#Scaling
num=['atemp','hum','windspeed','cnt']
dt_test[num]=scaler.fit_transform(dt_test[num])


# In[124]:


# X and y set creation for testing
y_test = dt_test.pop('cnt')
X_test = dt_test


# In[125]:


X_train_final = X_train_rfe10.drop(['const'], axis=1)


# In[127]:


# Prediction using our new model
X_test_final = X_test[X_train_final.columns]

#Constant variable
X_test_final = sm.add_constant(X_test_final)


# In[128]:


X_train_rfe10.columns


# In[129]:


#Prediction
y_pred = lmJ.predict(X_test_final)


# In[131]:


# Predicted data v/s Test data : Plotting
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('Y Test v/s Y Pred', fontsize = 25)
plt.xlabel('Y_Test', fontsize = 20)
plt.ylabel('Y_Pred', fontsize = 20)


# In[132]:


# Hence proved that the selected model is best model to judge the demand. 
# Most values of actual and predicted value overlaps hence the model is sucessful one
# Model will be successfull in explaining the change in demand as well.


# In[134]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
np.sqrt(mean_squared_error(y_test,y_pred))


# In[135]:


# R Square Calculation
R_squared = r2_score(y_test,y_pred)
R_squared


# In[136]:


# Actual R Square value for our model is 81.4%


# In[137]:


X1 = X_train_final.iloc[:,0].values


# In[138]:


# Residuals plotting
fig = plt.figure()
plt.scatter(X1,R)
fig.suptitle('Independent Variables v/s Residuals', fontsize = 25)
plt.xlabel('Independent Variables', fontsize = 20)
plt.ylabel('Residuals', fontsize = 20)
plt.show()


# ## The error terms will be correlated to each other as the auto correlation will affect the regression. So there is a demand dependency when we look up different observation.

# In[139]:


X_train_final.head()


# In[140]:


print(X_train_rfe10.columns)


# In[141]:


print(lmJ.summary())


# # CONCLUSION
# 
# ## THE LINEAR REGRESSION EQUATION FOR THE BEST PREDICTED MODEL IS :
# ## cnt=(0.238 * yr) - (0.091*holiday)-(0.207*spring)-(0.062*Sunday)+(0.119* '5')+(0.153* '8')+(0.183* '9')+(0.147 * '10')-(0.413* Thunderstorm)+(0.104* '7')+(0.147* '6')-(0.081* Misty)
# 
# # Hence Demand of Bikes depends on the below variables:
# ## 'yr' 'holiday' 'spring' 'Sunday' '5' '8' '9' '10' 'Thunderstorm' '7' 'Misty'
# 
# ## Demand increase yearly and on the 5th 6th 7th 8th 9th and 10th month of the year
# ## Demand decreases at the time of holidays and Sundays 
# ## Demand decreases on spring season and on Thunderstorm and Misty climatic condition
# 
# # COMPANY IS RECOMMENDED TO MAKE STRATEGY TO INCREASE THE BUSINESS DURING HOLIDAYS , SPRING AND THUNDERSTORM/MISTY CLIMATIC CONDITION TO INCREASE THEIR OVERALL SALES. AND ALSO FOCUS AND  ON THE 1ST 2ND 3RD 4TH 11TH AND 12TH MONTH OF YEAR WHERE THERE IS LESS SALES.
# 

# In[ ]:




