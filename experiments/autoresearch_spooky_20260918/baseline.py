"""Fresh, common starting candidate. No historical recipe or fitted artifact."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train(data_dir, artifact_dir, budget_seconds, seed):
    frame=pd.read_csv(Path(data_dir)/'fit.csv')
    vectorizer=TfidfVectorizer(ngram_range=(1,1),max_features=40000,sublinear_tf=True)
    x=vectorizer.fit_transform(frame.text.fillna(''))
    model=LogisticRegression(C=1.0,max_iter=300,random_state=seed)
    model.fit(x,frame.author)
    joblib.dump((vectorizer,model),Path(artifact_dir)/'model.joblib')
    return {'architecture':'word_unigram_tfidf_logistic','C':1.0,'max_features':40000,'seed':seed}

def predict(artifact_dir,texts):
    vectorizer,model=joblib.load(Path(artifact_dir)/'model.joblib')
    p=model.predict_proba(vectorizer.transform(texts))
    return p[:,[list(model.classes_).index(c) for c in ['EAP','HPL','MWS']]]
