import association_rules as rules

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# Read in the training data from the file and load into a DataFrame
data = pd.read_csv('train.csv', dtype={'VisitNumber': np.int32,
                                       'Weekday': str,
                                       'Upc': str,
                                       'ScanCount': np.int16,
                                       'DepartmentDescription': str,
                                       'TripType': np.int8,
                                       'FinelineNumber': np.float32})
train_data, dev_data = train_test_split(data)

from association_rules import AssociationRules
ar = AssociationRules(10, 1)
ar.fit(dev_data)

ar.transform(4)