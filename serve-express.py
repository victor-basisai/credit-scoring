import pickle
from typing import List, Optional

from bedrock_client.bedrock.model import BaseModel


# Ordered list of model features
FEATURES = [ 
    'LIMIT_BAL',
    'SEX',
    'EDUCATION',
    'MARRIAGE',
    'AGE',
    'PAY_1',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6',
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6',
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]


class Model(BaseModel):
    def __init__(self, path: Optional[str] = None):
        '''
        Loads the model
        '''
        with open(path or "/artefact/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, request_json):
        '''
        Runs the prediction
        '''
        # Parse request_json into ordered list
        features = list()
        for col in FEATURES:
            features.append(request_json[col])
        
        # Return the result
        result = {
            "predicted_proba": self.model.predict_proba(np.array(features).reshape(1, -1))[:, 1].item()
        }
        return result