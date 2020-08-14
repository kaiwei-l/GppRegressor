# Machine-Learning-Predict-GPP-PI-F

## Method
### Goal
The purpose of this research is to use the method of machine learning to study what variables do we need in order to estimate gross primary production (GPP) in a given length of time. In particular, we want to study the assumption that GPP can be solely predicted by our model if given only NIRV (Near-Infrared Reflectance of Terrestrial Vegetation).

### Materials
The data sources that we prepared for the ML model are from two sites, AmeriFlux Network and EPIC satellite. The first set of datasets we obtained are from AmeriFlux. AmeriFlux Network is a network of sites that are monitoring CO2, water and energy flow in America. By downloading data from AmeriFlux, we obtained several datasets that contain the raw data of local meteorological data from several observatory sites across America. The second set of datasets that we obtained are from EPIC satellite. The datasets contain the EPIC reflectance data, which includes specific time and three bands reflectance. Each dataset in the set was collected when EPIC satellite was above the observatory site directly, i.e. each dataset of the EPIC dataset collection corresponds to one observatory site of AmeriFlux sites. Both datasets collections have timestamp recorded.

### Data Cleaning
Data cleaning was performed for both dataset collections. For the AmeriFlux dataset, we selected data started from 2015 and onward because EPIC satellite only has data after 2015. In the AmeriFlux dataset, it contains many detailed data fields but for this research, we only need a few of the data fields. For this research, we selected temperature, photosynthetic photon flux density (PPFD), vapour-pressure deficit (VPD) and gross primary production (GPP). For each of the parameters, I exclude the rows where those parameters have -9999. Then according to the longitude of the sites, we changed the timestamp of those sites from local time to UTC time to match with the EPIC satellite timestamp. 

For the EPIC satellite dataset collection, first we cleared all the rows with null values. Then, we calculated the Near-Infrared Reflectance of Terrestrial Vegetation (NIRV) value for each row in the dataset by using blue and red reflectance data. 

Last step of the data cleaning process is to for each site, we combined the rows with matching timestamp so that now each site has a new value, i.e. NIRV value.

### Model Construction
We constructed a random forest regressor to estimate GPP given several parameters. To find the best parameters to tune the regressor, we employed grid search algorithm to find the best fit parameters. Then we split the data randomly into 20% and 80% and make 80% of the data into training set and 20% of the data into test set. To evaluate the model performance, we assessed the RMSE (Root mean square deviation) and R-squared score outputted from the trained model by feeding test set data. 

### Experiments Design
In order to test if our model can achieve decent performance by just using NIRV to estimate GPP, we designed three set of experiments. The first set of experiments used NIRV, temperature, PPFD and VPD. The second set of experiments used NIRV and PPFD. The last set of experiments used NIRV only. After finishing training the models, we feed them the testing dataset and assess the performance by plotting graph of the estimated data points against the true data points. Moreover, we evaluated the performance by the RMSE and R-Squared score values outputted from the models. 
