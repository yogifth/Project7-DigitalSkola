from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

# Models
class Applicant(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int
    Property_Area: str
    Loan_Status: str

# Import
model_lr = joblib.load("model_lr.joblib")

# Routes
@app.get("/")
def home():
    return{'message': 'Hello World'}

@app.post("/predict")
def predict(data: Applicant):
    inp_data = data.dict()
    header = []
    value = []
    for key, val in inp_data.items():
        header.append(key)
        value.append(val)
    
    inp = pd.DataFrame([value], columns=header)
    result = model_lr.predict(inp)
    return{"result": result}