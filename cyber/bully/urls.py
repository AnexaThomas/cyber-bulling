from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about', views.about, name="about"),
    path('contact', views.contact, name='contact'),
    path('login', views.login, name='login'),
    path('register', views.register, name='register'),
    path('logout', views.logout, name="logout"),
    path('prediction', views.prediction, name='prediction'),
    path('pred', views.pred, name="pred"),
    path('result', views.result, name="result"),
    path('comp', views.comppage, name="comppage"),
    path('fb', views.fb, name="fb"),
    path('viewcomplaint/', views.customerviewcomp, name='customerviewcomp'),
    path('forgot-password', views.forgotPassword, name = 'forgot-password'),
    path('update-password', views.updatePassword, name = 'update-password')

]
