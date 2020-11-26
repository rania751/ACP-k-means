from django.contrib import admin
from django.urls import path ,re_path , include
from .views import *

   
urlpatterns = [
	 path('document/',model_form_upload, name='document'),
     path('data/',dataView, name='data'),
     path('recherche/',search_data, name='search_data'),
     path('affichage/',dataView, name='affichage'),
     path('dataCR/',data_Centrage_Réduction_View, name='dataCR'),
     path('cluster/',clusterView, name='cluster'),
     #path('affichage_p/',affichage_p, name='affichage_p'),
     #path('affichage_cluster/',affichage_cluster, name='affichage_cluster'),
     path('pca/',pcaView, name='pca'),
     path('display_scree_plot/',display_scree_plot, name='display_scree_plot'),
     path('display_circles/',display_circles, name='display_circles'),
     path('display_factorial_planes/',display_factorial_planes, name='display_factorial_planes'),

]