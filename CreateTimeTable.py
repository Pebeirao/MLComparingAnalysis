
import pandas as pd 
import numpy as np 
import time
from sklearn.linear_model import LogisticRegression

# Read Attractness Attributes Table

AttractnessTable = pd.read_csv('archive/list_attr_celeba.csv')


predictorDataSet =  AttractnessTable.drop(['image_id', 'Attractive'], axis=1)
targetValue = AttractnessTable['Attractive']


# colIx of the values used to train the model 
colIx = []
# dictionary with all the values
coldict = {}  

for col in predictorDataSet.columns:
    #for each col in the Dataset add the col in the col index to use to train
    colIx.append(str(col))
    # list of time values
    values = []
    
    
    for ter in range(1,4):
        #for each 1/3 of the records to predict
        meanvalue = []
        for x in range(0, 100):
            # Run the predictor 100 values to have a mean value 

            predictorData = predictorDataSet.loc[0:ter*(len(predictorDataSet)/3),colIx]
            target = targetValue.loc[0:ter*(len(targetValue)/3)]
            
            start = time.time()
            clf = LogisticRegression(random_state=0).fit(predictorData, target)
            end = time.time()
            # print(end - start)
            meanvalue.append(end - start)

        values.append(sum(meanvalue)/100)
        
    coldict[len(colIx)] = values
    
AttTimeResultsTable = pd.Dataframe.from_dict(coldict, orient='index')      
AttTimeResultsTable.to_csv("AttTimeResultsTable.csv")
    




