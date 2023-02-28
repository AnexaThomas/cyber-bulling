from django.shortcuts import render, redirect
from .models import *
from django.contrib import messages
from django.contrib.auth.models import User, auth
import numpy as np
from sklearn.metrics import accuracy_score
from .models import feed
from django.http import HttpResponse
import pandas as pd
import re
from sklearn.svm import SVC
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from django.conf import settings
from django.contrib import messages
import pickle
from .forms import complaintForm

import nltk
nltk.download('stopwords')


# Create your views here.
def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


def pred(request):
    return render(request, 'pred.html')


def login(request):
    if request.method == "POST":

        try:
            Userdetails = user.objects.get(email=request.POST['email'], password=request.POST['password'])
            print("Username=", Userdetails)
            request.session['id'] = Userdetails.id
            print(request.session['id'])
            return render(request, 'prediction.html')
        except user.DoesNotExist as e:
            messages.success(request, 'Username/Password Invalid...!')
    return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST['email']
        password = request.POST['password']
        rpwd = request.POST['repeatpassword']


        emailExist = user.objects.filter(email = email)

        if(emailExist):
            messages.success(request,"E-mail Id already exist...!")
            return render(request,'register.html')
        else:
            user(name=name, email=email, password=password, rpwd=rpwd).save()
            messages.success(request, 'The New User ' + request.POST['name'] + " is saved Successfully...!")
            return redirect('/login')
    else:
        return render(request, 'register.html')
def fb(request):
    if request.method=='POST':
        name=request.POST['name']
        email=request.POST['email']
        message=request.POST['message']

        feed(name=name,email=email,message=message).save()
        messages.success(request,"Feedback sended Successfully...!")
        return render(request,'fb.html')
    
    else:
        return render(request,'fb.html')



def comppage(request):
    os.getcwd()
    print(os.getcwd())
    form = complaintForm()
    if request.method == 'POST':
        userid = request.POST['userid']
        email = request.POST['email']
        address = request.POST['address']
        phone = request.POST['phone']
        status = request.POST['status']
        social_media = request.POST['social_media']
        messager_id = request.POST['messager_id']
        message = request.FILES['message']
        print(message)
        # print('ssss', userid)
        messages.success(request, ' Complaint sended Successfully...!')
        Complaint(complainer=userid, email=email, address=address, phone=phone, status=status, social_media=social_media,messager_id=messager_id, message=message).save()

        lastId = Complaint.objects.latest('id')

        comp = Complaint.objects.filter(id=lastId.id)

        pp = os.getcwd()

        path = r'{}\media\documents'.format(pp)

        print('---------------------------------------------',path)

        new_list = []
        for x in comp:
            print(x.message)
            de = str(x.message)
            finalDoc = os.path.basename(de)

            append_str = 'documents/'
            for root, dirs, files in os.walk(path):
                for file in files:
                    if finalDoc == file:
                        with open(os.path.join(root, file), 'r') as f:
                            text = f.read()
                            print('ggggg', text)
                            new_list.append(text)
                            print(x.message)
                            Complaint.objects.filter(id=lastId.id).update(message=text)


                    else:
                        print('failed')



    else:
        form = complaintForm()

    return render(request, 'comp.html', {'form': form})



def customerviewcomp(request):
    try:
        val=request.session['id']
        # print(val)
    except:
        print("snsmnmsnm")
    # args = request.user.complainer_id
    # comp = Complaint.objects.filter(complainer=args)
    comp=Complaint.objects.filter(complainer=val)
    
    
    
    return render(request, 'viewcomplaint.html',{'comp':comp})


def logout(request):
    try:
        del request.session['email']
    except:
        return render(request, 'index.html')
    return render(request, 'index.html')


def result(request):
    return render(request, "result.html")


def prediction(request):
    textfile = request.POST.get('file')
    print('pp', textfile)
    fileData = request.FILES['file'].read()
    print('ddddddddddddddddddddddddddddddddddddd',fileData)

    csv_filepath = textfile
    if textfile:
        df = pd.read_csv(csv_filepath)

        csv_filepath = textfile
    try:
        rrr = pd.DataFrame(df)

        print('sssss',rrr.iloc[:1])
    # Add your other code to manipulate the dataframe read from the csv here
    except BaseException as exception:
        print(f"An exception occurred: {exception}")
    
    

    value = ''

    # if request.method == 'POST':

    def preprocess_tweet(tweet):
        # Remove words other than alphabets.
        row = re.sub("[^A-Za-z ]", "", tweet).lower()

        # Tokenize words.
        words = word_tokenize(row)

        # Remove stop words.
        english_stops = set(stopwords.words('english'))

        # Remove un-necessary words.
        characters_to_remove = ["''", '``', "rt", "https", "’", "“", "”", "\u200b", "--", "n't", "'s", "...", "//t.c"]
        clean_words = [word for word in words if word not in english_stops and word not in characters_to_remove]

        # Lematise words.
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]

        return " ".join(lemma_list)
    df = pd.read_csv('static/public_data_labeled.csv')
    df['Processed_Tweet'] = df['full_text'].map(preprocess_tweet)

    textfile = fileData

    decodedText = textfile.decode("utf-8")
    print('^^^^^^', decodedText)

    print('%%%%%%', type(textfile))

    cv = CountVectorizer(max_features=1500)


    X = cv.fit_transform(df['Processed_Tweet']).toarray()

    # Label encode.
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Logistic Regression.
    lr = LogisticRegression(random_state=0)

    # Train classifier.
    lr.fit(X_train, y_train)

    # Predict on train set.
    y_train_pred = lr.predict(X_train)

    # Predict on test set.
    y_test_pred = lr.predict(X_test)

    # save the model to disk

    # f = 'static/cybermodel.sav'
    # pickle.dump(lr, open(f, 'wb'))
    # print("Model saved")

    # Creating our training model.

    csv_filepath = decodedText
    # if textfile:
    # df = pd.read_csv(csv_filepath)
    csv_filepath = decodedText
    # try:
    test1 = [preprocess_tweet(decodedText)]
    test2 = cv.transform(test1)
    model = pickle.load(open("static/cybermodel.sav", "rb"))
    prediction = model.predict(test2)

    print('%%%%%%%%', prediction)
   

    if int(prediction[0]) == 1:
        value = 'Offensive'

    elif int(prediction[0]) == 0:
        value = "Non-offensive"
    x_train_prediction=model.predict(X_train)
    training_data_accuracy=accuracy_score(x_train_prediction,y_train)
    print("Accuracy on Training data:",training_data_accuracy)
    x_test_prediction=model.predict(X_test)
    test_data_accuracy=accuracy_score(x_test_prediction,y_test)
    print("Accuracy on Test data:",test_data_accuracy)
    return render(request,
                  'result.html',
                  {
                      'context': value,
                      'title': 'Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'background': 'bg-primary text-white'
                  })


def pred(request):
    if request.method == "POST":
        return render(request, 'chart.html')
    return render(request, 'pred.html')


def forgotPassword(request):
    return render (request,'forgot-password.html')

def updatePassword(request):
    email = request.POST['email']
    password = request.POST['password']
    
    checkAccountExistOrNot = user.objects.filter(email = email)
    if checkAccountExistOrNot:
        user.objects.filter(email = email).update(password = password)
        messages.success(request,"Password updated Successfully...!")
        return render(request,'login.html')
    else:
        messages.success(request,"No account found on this E-mail Id...!")
        return render (request,'forgot-password.html')
    