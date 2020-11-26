from django.contrib				 import admin
from django.urls 				import path ,re_path , include
from django.contrib.auth.views import LoginView ,LogoutView , PasswordResetView 
from .views 					import *
from Accounts.views             import *
from Analyse.views             import *

   
urlpatterns = [
     #path('login/', LoginView.as_view(), name='login'),
     path('register/', registerView, name='register'),
     path('Dashboard/', DashboardView, name='Dashboard'),
     path('logout/', LogoutView.as_view(next_page="login"), name='logout'),

]