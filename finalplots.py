# General libraries.
import pandas as pd
import numpy as np

def plots():
    # Read in the training data from the file and load into a DataFrame
    df = pd.read_csv('train.csv')

    # sort by VisitNumber to ensure everything is in the right order  
    df = df.sort_values('VisitNumber')
    df = df.reset_index()
    del df['index']

    # Check for null values
    p6 = df.isnull().sum(axis=0)

    # Replace nulls will 'Unknown' product categories
    df['Upc'] = df['Upc'].fillna('UnknownUpc')
    df['DepartmentDescription'] = df['DepartmentDescription'].fillna('UnknownDD')
    df['FinelineNumber'] = df['FinelineNumber'].fillna('UnknownFN')

    # Convert Weekday from a text variable to a numerical variable
    df['Weekday'] = df['Weekday'].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})

    # Plot histogram of ScanCount
    p2 = df.groupby(['ScanCount'], as_index=False)['TripType'].count()

    # Plot histogram of ScanCount where ScanCount>2
    p3 = df[df['ScanCount']>2].groupby(['ScanCount'], as_index=False)['TripType'].count()
    
    # Aggregate number of products by VisitNumber, i.e. count TripType
    wd = df.groupby(['VisitNumber','Weekday'], as_index=False)['TripType'].count()
    wd = wd.rename(columns={'TripType': 'NumProducts'})

    # Plot Weekday histogram
    p1 = wd.groupby(['Weekday'], as_index=False)['VisitNumber'].count()
    
    # Aggregate number of items by VisitNumber, i.e sum ScanCount
    tt = df.groupby(['VisitNumber','TripType'], as_index=False)['ScanCount'].sum()
    tt = tt.rename(columns={'ScanCount': 'NumItems'})

    # Sort by VisitNumber to get and ordered target list of TripType and row names of VisitNumber
    tt['NumProducts'] = wd.NumProducts
    tt.sort_values('VisitNumber')

    p4 = tt.groupby(['NumItems'], as_index=False)['TripType'].count()

    p5 = tt.groupby(['NumProducts'], as_index=False)['TripType'].count()
    
    # Plot histogram of TripType
    p7 = tt.groupby(['TripType'], as_index=False)['NumItems'].count()
    
    return p1, p2, p3, p4, p5, p6, p7
