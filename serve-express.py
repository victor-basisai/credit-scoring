import json
import pickle
from bedrock_client.bedrock.model import BaseModel
from typing import Any, AnyStr, BinaryIO, List, Mapping, Optional, Union


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

    def __init__(self):
        with open("/artefact/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, features: List[List[float]]) -> List[float]:
        return self.model.predict_proba(features)[:, 0].tolist()

    # Optional - Pre-process
    def pre_process(
            self, http_body: AnyStr, files: Optional[Mapping[str, BinaryIO]] = None
        ) -> List[List[float]]:
        
        # Input is a JSON
        samples = json.loads(http_body)

        # Parse JSON into ordered list
        features = list()
        for col in FEATURES:
            features.append(samples[col])
        return [[float(x) for x in s] for s in features]

    # Optional - Post-process
    def post_process(
            self, score: Union[List[float], List[Mapping[str, float]]], prediction_id: str
        ) -> Union[AnyStr, Mapping[str, Any]]:

        return {"result": score, "prediction_id": prediction_id}