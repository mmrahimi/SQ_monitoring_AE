import pandas as pd
import numpy as np 
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler


class getting_data_ready:
    
    def __init__(self, data, test_size):
        self.data = data
        self.test_size = test_size
        self.data_resampled = {}
        self.x_train_resampled = []
        self.test_data = {}
        
    def splitting_data(self):
        '''
        A dropout layer for sparse input data, note that this layer
        can not be applied to the output of SparseInputDenseLayer
        because the output of SparseInputDenseLayer is dense.
        '''
        data_random = self.data.sample(frac=1)

        X = data_random[['guid','txt']].to_numpy()
        Y= data_random[['Safety','CleanlinessView','Information','Service','Comfort','PersonnelCard','Additional']].to_numpy()

        self.X_train, self.y_train, self.X_test, self.y_test = iterative_train_test_split(X, Y, test_size = self.test_size)
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def to_table(self, X,y):
        tbl = {}
        tbl["guid"] = []
        tbl["txt"] = []
        tbl['Safety']= []
        tbl['CleanlinessView']= []
        tbl['Information']= []
        tbl['Service']= []
        tbl['Comfort']= []
        tbl['PersonnelCard']= []
        tbl['Additional']= []
        
        for ind,row in enumerate(y):
            tbl['guid'].append(X[ind][0])
            tbl['txt'].append(X[ind][1])
            tbl['Safety'].append(row[0])
            tbl['CleanlinessView'].append(row[1])
            tbl['Information'].append(row[2])
            tbl['Service'].append(row[3])
            tbl['Comfort'].append(row[4])
            tbl['PersonnelCard'].append(row[5])
            tbl['Additional'].append(row[6])

        return pd.DataFrame.from_dict(tbl)
    
    def resampling_data(self, X, y):

        # Import a dataset with X and multi-label y
        lp = LabelPowerset()
        ros = RandomOverSampler(random_state=42)

        # Applies the above stated multi-label (ML) to multi-class (MC) transformation.
        yt = lp.transform(y)
        X_resampled, y_resampled = ros.fit_sample(X, yt)
        # Inverts the ML-MC transformation to recreate the ML set
        y_resampled = lp.inverse_transform(y_resampled)

        return X_resampled, y_resampled
    
    
    def resampled_to_table(self,X_resampled, y_resampled):
        self.data_resampled = self.to_table(X_resampled, y_resampled.toarray())
        self.x_train_resampled = self.data_resampled.sample(frac=1)
        return self.x_train_resampled
    
    def report_on_resampled_classes(self):
        print('Safety: ',np.sum(self.x_train_resampled['Safety']))
        print('CleanlinessView: ',np.sum(self.x_train_resampled['CleanlinessView']))
        print('Information: ',np.sum(self.x_train_resampled['Information']))
        print('Service: ',np.sum(self.x_train_resampled['Service']))
        print('Comfort: ',np.sum(self.x_train_resampled['Comfort']))
        print('PersonnelCard: ',np.sum(self.x_train_resampled['PersonnelCard']))
        print('Additional: ',np.sum(self.x_train_resampled['Additional']))