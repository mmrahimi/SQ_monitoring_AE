import csv
import os
import pandas as pd
import numpy as np

class data_from_csv:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.data["guid"] = []
        self.data["txt"] = []
        self.data['Safety']= []
        self.data['CleanlinessView']= []
        self.data['Information']= []
        self.data['Service']= []
        self.data['Comfort']= []
        self.data['PersonnelCard']= []
        self.data['Additional']= []
        self.read = False
    
    def is_in_class(self, classes, columnID):
        classes = classes.split(',')
        classes = [int(x.strip()) for x in classes if len(x.strip())>0]
        if columnID in classes:
            return 1
        else:
            return 0
    
    def read_csv(self):
        with open(self.path, newline='', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                self.data['guid'].append(row[0])
                self.data['txt'].append(row[2])
                self.data['Safety'].append(self.is_in_class(row[3],0))
                self.data['CleanlinessView'].append(self.is_in_class(row[3],1))
                self.data['Information'].append(self.is_in_class(row[3],2))
                self.data['Service'].append(self.is_in_class(row[3],3))
                self.data['Comfort'].append(self.is_in_class(row[3],4))
                self.data['PersonnelCard'].append(self.is_in_class(row[3],5))
                self.data['Additional'].append(self.is_in_class(row[3],6))
        
        self.read = True    
        return pd.DataFrame.from_dict(self.data)
    
    def report_on_classes(self):
        if (self.read):
            data = pd.DataFrame.from_dict(self.data)
            print('Safety: ',np.sum(data['Safety']))
            print('CleanlinessView: ',np.sum(data['CleanlinessView']))
            print('Information: ',np.sum(data['Information']))
            print('Service: ',np.sum(data['Service']))
            print('Comfort: ',np.sum(data['Comfort']))
            print('PersonnelCard: ',np.sum(data['PersonnelCard']))
            print('Additional: ',np.sum(data['Additional']))
        else:
            print("You need to first run 'read_csv' method")