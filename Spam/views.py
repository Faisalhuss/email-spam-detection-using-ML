from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib
import numpy as np

model2 = joblib.load(os.path.dirname(__file__) + "\\svc_model\\mySVCModel.pkl")
model1 = joblib.load(os.path.dirname(__file__) + "\\random_forest\\myModel.pkl")

model3 = joblib.load(os.path.dirname(__file__) + "\\bagging\\best_model.pkl")



loaded_model = joblib.load(os.path.dirname(__file__) + '\\ada_boost\\adaboost_model_balanced.pkl')
loaded_tfidf = joblib.load(os.path.dirname(__file__) + '\\ada_boost\\tfidf_vectorizer.pkl')

loaded_model1 = joblib.load(os.path.dirname(__file__) + '\\nb_model\\best_model_nb.pkl')
loaded_tfidf1 = joblib.load(os.path.dirname(__file__) + '\\nb_model\\tfidf_vectorizer_nb.pkl')

def index(request):
    return render(request, 'index.html')

def checkSpam(request):
    if request.method == "POST":
        algo = request.POST.get("algo")
        rawData = request.POST.get('rawdata')
        param = {}
        if not rawData.strip():
            param = {"answer": "No input"}

        elif algo == "Algo-1":
            finalAns = model1.predict([rawData])[0]
            param = {"answer": finalAns}
        
        elif algo == "Algo-2":
            finalAns = model2.predict([rawData])[0]
            param = {"answer": finalAns}

        elif algo == "Algo-3":    
        
            vector_input = loaded_tfidf1.transform([rawData])
            result = loaded_model1.predict(vector_input)
            param = {"answer": result}

        elif algo == "Algo-4":
            finalAns = model3.predict([rawData])[0]
            param = {"answer": finalAns}

        elif algo == "Algo-5":
        
            text_vectorized = loaded_tfidf.transform([rawData])

            prediction = loaded_model.predict(text_vectorized)
            param = {"answer": prediction}

        else:
            param = {"answer": "No algorithm selected"}
        return render(request, 'output.html', param)
    else:
        return render(request, 'index.html')
