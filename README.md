## Predict Bike Sharing Demand with AutoGluon Solution

This project aims to predict bike sharing demand using AutoGluon in a bike rental service company, if given features: DateTime, season, holiday,	working day,	weather, temperature, humidity,
wind speed, and user type. The reason why AutoGluon is utilized is simply because it's easy to use and easy to extend AutoML.

## Approach
1. **Data Cleaning**
   
It involves ensuring the dataset is free from missing or inconsistent information.

2. **Initial Model Training**

An initial model was trained using AutoGluon. As the dataset was already clean and divided into training and testing sets, we directly trained the model and generated predictions from the test set.
To match the submission format, a new CSV file containing "datetime" and "count" columns was created. These columns must correspond exactly to those in the submission.csv file.
Additionally, negative predicted values were replaced with zero to avoid submission rejection.

3. **Exploratory Data Analysis (EDA)**
   
To improve the RMSE score, exploratory data analysis was conducted. Correlation analysis, histograms, and line charts were used to examine relationships between features, understand data distribution, and identify seasonal patterns.
New features were created by extracting hour, day, month, and year from the "datetime" column. Categorical features, such as season and weather, were encoded.

4. **Model Retraining**
   
The model was retrained and evaluated.

5. **Hyperparameter Tuning**
   
Hyperparameter tuning was performed three times by adjusting AutoGluon's parameters and model selection to achieve satisfactory results.

## Result

### Initial training
The top-ranked model is WeightedEnsemble_L3, with an RMSE score of 139.610795 in the train set.
Whereas, from the 19 trained models the worst model is KNeighborsDist_BAG_L1 with an RMSE score of 169.433436. The top-ranked model produces an RMSE score of 1.32509 in the test set.

### EDA
I explore the numerical features distribution in the train.csv
It was found that:
- Seasons 1, 2, 3, and 4 have equal popularity as shown by the identical count in the bar plot.
- Bike demand tends to be higher on holidays compared to weekdays.
- Temperature follows a normal distribution, while attempts and humidity are skewed to the left and windspeed to the right.
- I observe higher bike demand when the temperature is between 10 and 32 degrees, attempts are between 10 and 38, humidity is between 40 and 85%, and windspeed is around 0 to 20 km/h compared to other weather conditions.
- The hour feature is skewed to the right, with over 1750 users renting bikes at hour 0. Similarly, the day feature is skewed to the right, and more than 1750 users rent bikes on day 0.
- An interesting pattern was found in the month feature, with consistent demand from month 1 to month 22.
- The correlation map shows a low correlation (correlation score of 0.48) between season and datetime. Most independent variables have a weak correlation with the dependent variable, with correlation scores between 0.3 and 0.5. Examples include count-temperature, count-attempts, and count-humidity.
- Time series analysis shows that June 2011 and 2012 were peak periods for bike share demand.
- Daily demand fluctuates, with the lowest demand on day 1 and the highest on day 16 (note: 1 and 16 are encoded day values).

### Changes after adding new features
RMSE score was reduced from 1.32509 to 0.48189 after adding additional features, it improved model performance by 84.32%.
I think the reason is because the model was given more input, hence it provides more features to be learned and the model can better recognize patterns when there will be high and low demand.

### 3 round: Hyperparameter Tuning
In the first experiment, using NeuralNetworkFastAI and CaTBoost, the model performed poorly. The RMSE error rose to 1.19842, almost the same as when no additional features were used. The reason for this is unclear, but there might have been a configuration error.
However, in the second hyperparameter configuration, using GBM, RF, CAT, and XT, the model improved. The RMSE score decreased to 0.46699. In the third configuration, using CAT, XT, and XGB with a wider search space than in the second experiment, the results were disappointing. The RMSE score remained the same as in the first experiment.
Based on these results, tuning only the best model from the baseline would be more effective than tuning all models, including the worst-performing ones. This would also save time, memory, CPU, and energy.

### Recap
| **model_name** | **parameter_1** | **parameter_2** | **parameter_3** | **score**|
|--|--|--|--|--|
|initial|original columns|time_limit=600|presets=best_quality|1.32509|
|add_features|extract year, month, day, hour from datetime|time_limit=600|presets=best_quality|0.48189|
|hpo_1|added features|hyperpar_tune={searcher:random,max_tune_time:3600,scheduler:local,num_trials:4}|Model: CAT, FASTAI|1.19842|
|hpo_2|added features|hyperpar_tune=same like hpo_1|Model: GBM, RF, CAT, XT|0.46699|
|hpo_3|added features|hyperpar_tune=same like hpo_2, change max_tune_time to 4800|same models just like in hpo_2, adding more search space in each model|1.19842|

## Summary
- Ensure that the data is clean and free of duplicates, inconsistencies, missing values before proceeding to exploratory data analysis and subsequent steps.
- Extracting additional features from existing data could significantly improve model performance, particularly when working with time series data.
- Using AutoGluon for Kaggle competitions helped save time compared to training individual models. However, tuning the model using AutoGluon was challenging due to insufficient details in the documentation about model parameters.
- Ultimately, our training strategy will depend on our goal. For example, training a model to achieve the lowest error rate will differ from training a model for deployment, impacting preprocessing and model selection.
- The trial results showed that the lowest RMSE score on the test set was obtained by using additional features and tuning the baseline models: GBM, RF, CAT, and XT.

## Future work
Spend more time on Feature Engineering, and also only Fine-tune the top 1-3 model from the baseline. In feature engineering, I probably will add 2 features: weekday and weekend. Yeah, it will be similar to the available features: holiday and working day.
I can use 4 of them, drop the available feature then change with the new one, or just use weekday, weekend, and holiday. Lastly, in model fine-tuning, I would only fine-tune the following model: CatBoost, ExtraTree, LightGBM (I take references from add_features model performance).
I'll focus on playing around with each model search space.
