from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.feature_selection import SelectKBest, f_regression


path_X_train="project/given_data/spatiotemporal_trn_data.csv"                   #### enter path for file which has training features
path_X_test="project/given_data/spatiotemporal_tst_data (3).csv"                 #### Enter path for file which has testing features
path_y_train="project/given_data/spatiotemporal_trn_targets.csv"

y=pd.read_csv(path_y_train,header=None)
y.columns = ['Column1', 'Value']
y_train=y['Value']


X_train=pd.read_csv(path_X_train,dtype=object)
X_test=pd.read_csv(path_X_test,dtype=object)

def transform_X(dataset):                                               ##### Function to transform both training and test set

    weather=dataset
    weather['DATE'] = pd.to_datetime(weather['DATE'])   
    weather= weather.sort_values(by=['NAME','DATE'])
    weather.reset_index(drop=True, inplace=True)

    weather['hour'] = weather['DATE'].dt.hour
    weather['month'] = weather['DATE'].dt.month
    weather=weather[["hour","month","ELEVATION","REPORT_TYPE","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyPresentWeatherType","HourlyRelativeHumidity","HourlySeaLevelPressure","HourlyWindDirection","HourlyWindSpeed"]]

    pattern = r'^(\d+)s$'

# Loop through the columns you want to transform
    columns_to_transform = ['HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed']

    for column in columns_to_transform:
        weather[column] = weather[column].apply(lambda x: re.sub(pattern, r'\1', str(x)) if isinstance(x, str) else x)
        weather[column] = pd.to_numeric(weather[column], errors='coerce')

    weather['HourlyWindDirection'] = weather['HourlyWindDirection'].replace('VRB', 0)
    weather[['ELEVATION','HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed']] = weather[['ELEVATION','HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed']].astype(float)

    transformer = ColumnTransformer(transformers=[
        ('tnf1',OneHotEncoder(sparse=False,drop='first',handle_unknown='infrequent_if_exist',min_frequency=2000),['REPORT_TYPE']),
        ('tnf2',OneHotEncoder(sparse=False,drop='first',handle_unknown='infrequent_if_exist',min_frequency=1000),['HourlyPresentWeatherType'])
    ],remainder='passthrough')


    weather_1=transformer.fit_transform(weather)
    transformed_df = pd.DataFrame(weather_1)

    imputer = SimpleImputer(strategy='median')
    transformed_df=imputer.fit_transform(transformed_df)

    transformed_df = pd.DataFrame(transformed_df)

    return transformed_df

##### For target variable
########################################## RUN THIS FUNCTION FOR Y_TEST TOO,BEFORE EVALUATING #############
def correct_y(targets):
    target=pd.DataFrame(y_train)
    pattern = r'^(\d+)V$'   ## Identifying the anomaly
    target['Value'] = target['Value'].apply(lambda x: re.sub(pattern, r'\1', str(x)) if isinstance(x, str) else x)
    # Convert the 'visibility' to integers using pd.to_numeric, and coerce to handle NaN values
    target['Value'] = pd.to_numeric(target['Value'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    transformed_target=imputer.fit_transform(target)
    filtered_y_train=transformed_target

    return filtered_y_train

transformed_X_train=transform_X(X_train)
transformed_X_test=transform_X(X_test)
filtered_y_train=correct_y(y_train)
##### Running the best performing model ##############from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.feature_selection import SelectKBest, f_regression

# Assuming X, y are your features and target
selector = SelectKBest(score_func=f_regression, k=18)  # Select the desired number of features
transformed_X_train= selector.fit_transform(transformed_X_train, filtered_y_train)

regressor = RandomForestRegressor(max_features=None,random_state=0)
# Train the model
regressor.fit(transformed_X_train, filtered_y_train)
# Make predictions on the test set
y_pred = regressor.predict(transformed_X_test)
visibility = pd.DataFrame(data=y_pred, columns=['Prediction'])
visibility.to_csv('visibility_21290.csv', index=False)
print(y_pred)

'''######################################################
###### CODE USED WHILE DEVELOPING THE MODEL###########
######################################################


######## Impporting both training and test data and comcantating ########
path_x="given_data/spatiotemporal_trn_data.csv"
path_target="given_data/spatiotemporal_trn_targets.csv"
weather=pd.read_csv(path_x,dtype=object)
y=pd.read_csv(path_target,header=None)
y.columns = ['Column1', 'Value']
weather['visibility']=y['Value']                                                         


####### Sorting values ##############
weather['DATE'] = pd.to_datetime(weather['DATE'])
weather= weather.sort_values(by=['NAME','DATE'])
weather.reset_index(drop=True, inplace=True)

############ Missing Values ##########
#pd.set_option('display.max_rows', None)     # comment this out to see all rows
nan_percentage = (weather.isna().mean() * 100).round(2)
print("Percentage of NaN values for each column:")
print(nan_percentage)
pd.reset_option('display.max_rows')

#########Extracting relevant features(Day and Month)######
eather['hour'] = weather['DATE'].dt.hour          #### Hour of the date in 24 hr format ####
weather['month'] = weather['DATE'].dt.month        #### Month of the year in number ########
weather=weather[["hour","month","ELEVATION","REPORT_TYPE","HourlyDewPointTemperature","HourlyDryBulbTemperature","HourlyPresentWeatherType","HourlyRelativeHumidity","HourlySeaLevelPressure","HourlyWindDirection","HourlyWindSpeed","visibility"]]

######### Removing character anamoly ########

##### For target variable
pattern = r'^(\d+)V$'   ## Identifying the anomaly
weather['visibility'] = weather['visibility'].apply(lambda x: re.sub(pattern, r'\1', str(x)) if isinstance(x, str) else x)
# Convert the 'visibility' to integers using pd.to_numeric, and coerce to handle NaN values
weather['visibility'] = pd.to_numeric(weather['visibility'], errors='coerce')

##### For features  
pattern = r'^(\d+)s$'
# Loop through the columns you want to transform
columns_to_transform = ['HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed']
for column in columns_to_transform:
    weather[column] = weather[column].apply(lambda x: re.sub(pattern, r'\1', str(x)) if isinstance(x, str) else x)
    # Convert the column to integers using pd.to_numeric, and coerce to handle NaN values
    weather[column] = pd.to_numeric(weather[column], errors='coerce')

weather['HourlyWindDirection'] = weather['HourlyWindDirection'].replace('VRB', 0)
weather[['ELEVATION','HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed','visibility']] = weather[['ELEVATION','HourlyDewPointTemperature', 'HourlyDryBulbTemperature','HourlyRelativeHumidity','HourlySeaLevelPressure','HourlyWindDirection','HourlyWindSpeed','visibility']].astype(float)

##### TARGET VARIABLE###############
min=weather['visibility'].min()
max= weather['visibility'].max()
print(f"Minimum visibility:- {min},Maximum visibility:- {max}" )
## Plotting distribution of target variable
bin_edges = np.arange(0, 101, 5)
weather['visibility'].plot.hist(bins=bin_edges)
plt.xlabel("Visibility")
plt.title("Distribution of Target Variable")
plt.xticks(bin_edges)
plt.savefig('visibility.png')
plt.show()

############## One Hot Encoding ###########

transformer = ColumnTransformer(transformers=[
    ('tnf1',OneHotEncoder(sparse=False,drop='first',handle_unknown='infrequent_if_exist',min_frequency=2000),['REPORT_TYPE']),
    ('tnf2',OneHotEncoder(sparse=False,drop='first',handle_unknown='infrequent_if_exist',min_frequency=1000),['HourlyPresentWeatherType'])
],remainder='passthrough')
weather_1=transformer.fit_transform(weather)
transformed_df = pd.DataFrame(weather_1)

########## Imputing Values #################
imputer = SimpleImputer(strategy='median')
transformed_df=imputer.fit_transform(transformed_df)
transformed_df = pd.DataFrame(transformed_df)
transformed_df.to_csv('table.csv', index=False)

##################################################################
########## GRID SEARCH WITHOUT SCALING ###########################
##################################################################


class regression():
     def __init__(self,path='table.csv',rgr_opt='lr',no_of_selected_features=None):
        self.path = path
        self.rgr_opt=rgr_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features)

# Selection of regression techniques
     def regression_pipeline(self):
    # AdaBoost
        if self.rgr_opt=='ab':
            print('\n\t### AdaBoost Regression ### \n')
            be1 = DecisionTreeRegressor(max_depth=10,ccp_alpha=0.02,random_state=0)
            be2 = Ridge(alpha=1.0,solver='lbfgs',positive=True)
            
            rgr = AdaBoostRegressor(n_estimators=100)
            rgr_parameters = {
            'rgr__estimator':(be1,be3,be3),
            'rgr__random_state':(0,10),
            }
    # Decision Tree
        elif self.rgr_opt=='dt':
            print('\n\t### Decision Tree ### \n')
            rgr = DecisionTreeRegressor(random_state=40)
            rgr_parameters = {
            'rgr__criterion':('squared_error','friedman_ms','absolute_error', 'poisson'),
            'rgr__max_depth':(30,None),
            'rgr__ccp_alpha':(0.009,0.00),
            }
    # Ridge Regression
        elif self.rgr_opt=='rg':
            print('\n\t### Ridge Regression ### \n')
            rgr = Ridge(alpha=1.0,positive=True)
            rgr_parameters = {
            'rgr__solver':('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'),
            }
    # Linear Regression
        elif self.rgr_opt=='lr':
            print('\n\t### Linear Regression ### \n')
            rgr = LinearRegression()
            rgr_parameters = {
            'rgr__positive':(True,False),
            }
    # Random Forest
        elif self.rgr_opt=='rf':
            print('\n\t ### Random Forest ### \n')
            rgr = RandomForestRegressor(max_features=None)
            rgr_parameters = {
            #'rgr__criterion':('squared_error','friedman_mse','poisson'),
            'rgr__n_estimators':(100,130,150),
            'rgr__max_depth':(50,70,None),
            'rgr__max_features':(1,5,10),
            }
    
        else:
            print('Select a valid classifier \n')
            sys.exit(0)
        return rgr,rgr_parameters

# Load the data
     def get_data(self):
    # Load the file using CSV Reader
        # fl=open(self.path+'winequality_white.csv',"r")
        # reader = list(csv.reader(fl,delimiter='\n'))
        # fl.close()
        # data=[]; labels=[];
        # for item in reader[1:]:
        #     item=''.join(item).split(';')
        #     labels.append(item[-1])
        #     data.append(item[:-1])
        # # labels=[int(''.join(item)) for item in labels]
        # data=np.asarray(data)

    # Load the file using Pandas
        reader=pd.read_csv('table.csv')
        data=reader.iloc[:, :-1]
        labels=reader.iloc[:,-1]

        # Training and Test Split
        training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels,
                                               test_size=0.3, random_state=42)

        return training_data, validation_data, training_cat, validation_cat

# Regression using the Gold Statndard after creating it from the raw text
     def regression(self):
   # Get the data
        training_data, validation_data, training_cat, validation_cat=self.get_data()

        rgr,rgr_parameters=self.regression_pipeline()
        pipeline = Pipeline([('rgr', rgr),])
        grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10)
        grid.fit(training_data,training_cat)
        rgr= grid.best_estimator_
        print('\n\n The best set of parameters of the pipiline are: ')
        print(rgr)
        joblib.dump(rgr, self.path+'training_model.joblib')
        predicted=rgr.predict(validation_data)


    # Regression report
        mse=mean_squared_error(validation_cat,predicted,squared=True)
        print ('\n MSE:\t'+str(mse))
        rmse=mean_squared_error(validation_cat,predicted,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(validation_cat,predicted,multioutput='variance_weighted')
        print ('\n R2-Score:\t'+str(r2))

warnings.filterwarnings("ignore")
rgr=regression( rgr_opt='knn',
               no_of_selected_features=18)

rgr.regression()

##################################################################
########## GRID SEARCH WITH SCALING ###########################
##################################################################

class regression():
     def __init__(self,path='table.csv',rgr_opt='lr',no_of_selected_features=None):
        self.path = path
        self.rgr_opt=rgr_opt
        self.no_of_selected_features=no_of_selected_features
        if self.no_of_selected_features!=None:
            self.no_of_selected_features=int(self.no_of_selected_features)

     def regression_pipeline(self):
         
    # Ridge Regression
         if self.rgr_opt=='rg':
            print('\n\t### Ridge Regression ### \n')
            rgr = Ridge(alpha=1.0,positive=True)
            rgr_parameters = {
            'rgr__solver':('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'),
            }
    # Linear Regression
         elif self.rgr_opt=='lr':
            print('\n\t### Linear Regression ### \n')
            rgr = LinearRegression()
            rgr_parameters = {
            'rgr__positive':(True,False),
            }

    # KNeighbors Regressor
         elif self.rgr_opt=='knn':
            print('\n\t### KNeighbors Regressor  ### \n')
            rgr = KNeighborsRegressor()
            rgr_parameters = {
            'rgr__n_neighbors':(3,5,8),
            'rgr__weights':('uniform', 'distance')
            }
         else:
            print('Select a valid classifier \n')
            sys.exit(0)
         return rgr,rgr_parameters

     def get_data(self):
    # Load the file using Pandas
        reader=pd.read_csv('table.csv')

        data=reader.iloc[:, :-1]
        labels=reader.iloc[:,-1]

        # Training and Test Split
        training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels,
                                               test_size=0.3, random_state=42)

        scaler1 = StandardScaler()
        scaler2 = RobustScaler()
        scaler3 = MinMaxScaler()
        # fit the scaler to the train set, it will learn the parameters
        scaler2.fit(training_data)

        # transform train and test sets
        training_data_scaled = scaler2.transform(training_data)
        validation_data_scaled = scaler2.transform(validation_data)

        return training_cat, validation_cat,training_data_scaled,validation_data_scaled

# Regression using the Gold Statndard after creating it from the raw text
     def regression(self):
   # Get the data
        training_cat,validation_cat,training_data_scaled,validation_data_scaled=self.get_data()

        rgr,rgr_parameters=self.regression_pipeline()
        pipeline = Pipeline([('rgr', rgr),])
        grid = GridSearchCV(pipeline,rgr_parameters,scoring='f1_macro',cv=10)
        grid.fit(training_data_scaled,training_cat)
        rgr= grid.best_estimator_
        print('\n\n The best set of parameters of the pipiline are: ')
        print(rgr)
        joblib.dump(rgr, self.path+'training_model.joblib')
        predicted=rgr.predict(validation_data_scaled)


    # Regression report
        mse=mean_squared_error(validation_cat,predicted,squared=True)
        print ('\n MSE:\t'+str(mse))
        rmse=mean_squared_error(validation_cat,predicted,squared=False)
        print ('\n RMSE:\t'+str(rmse))
        r2=r2_score(validation_cat,predicted,multioutput='variance_weighted')
        print ('\n R2-Score:\t'+str(r2)) 



import warnings

warnings.filterwarnings("ignore")
rgr=regression( rgr_opt='knn',
               no_of_selected_features=18)

rgr.regression()

#################################################################
#########  DEFAULT PARAMTERS WITHOUT SCLAING ####################
#################################################################

reader=pd.read_csv('table.csv')  
data=reader.iloc[:, :-1]
labels=reader.iloc[:, -1]
    
# Training and test split WITHOUT stratification        
training_data, validation_data, training_cat, validation_cat = train_test_split(data, labels, 
                                                test_size=0.30, random_state=42)

print('\n Training Data ')
training_cat=[x for x in training_cat]

print('\n Validation Data ')
validation_cat=[x for x in validation_cat]

# Regression
     
rgr1 = LinearRegression() 
rgr2 = Ridge(alpha=1.0,solver='lbfgs',positive=True) 
rgr4 = DecisionTreeRegressor(max_depth=12,ccp_alpha=0.02,random_state=10)
rgr5= RandomForestRegressor(max_features=None,random_state=0)                                     
rgr6= SVR()                                      
rgr7= KNeighborsRegressor()

rgr7.fit(training_data,training_cat)
predicted=rgr7.predict(validation_data)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))


#################################################################
####### DEFAULT PARAMTERS WITH SCALING ##########################
#################################################################

scaler2 = RobustScaler()

# fit the scaler to the train set, it will learn the parameters
scaler2.fit(training_data)

# transform train and test sets
training_data_scaled = scaler2.transform(training_data)
validation_data_scaled = scaler2.transform(validation_data)

rgr1 = LinearRegression()
rgr2 = Ridge(alpha=1.0,solver='lbfgs',positive=True)
rgr4 = KNeighborsRegressor(n_neighbors=5)
 

rgr4.fit(training_data_scaled,training_cat)
predicted=rgr4.predict(validation_data_scaled)

# Regression report
mse=mean_squared_error(validation_cat,predicted,squared=True)
print ('\n MSE:\t'+str(mse)) 
rmse=mean_squared_error(validation_cat,predicted,squared=False)
print ('\n RMSE:\t'+str(rmse))
r2=r2_score(validation_cat,predicted,multioutput='variance_weighted') 
print ('\n R2-Score:\t'+str(r2))'''