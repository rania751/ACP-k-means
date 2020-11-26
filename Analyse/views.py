from django.contrib.auth.decorators import login_required
import json 

import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
from .functions import *

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from Analyse.models import Document
from Analyse.forms import DocumentForm


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans


from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponseRedirect  
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.pyplot


import io
# Create your views here.


@login_required
def Dashboard(reqeust ):
	template_name = 'pages/Dashboard.html'
	return render(reqeust , template_name)

@login_required
def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('Dashboard')
    else:
        form = DocumentForm()
    return render(request, 'pages/model_form_upload.html', {'form': form})

donnees= None #Document.objects.get(description='data').document
data = None #pd.read_csv(donnees,decimal=".",index_col=0) 
n_clust = None # Nombre de clusters souhaités
X = None #data.values # préparation des données pour le clustering

n_comp=2
names = None #data.index # ou data.index pour avoir les intitulés
features = None #data.columns  
X_scaled=data
centroids=X_scaled
pca=data
dat = None
clusters = ""

def service(request):

    Document_options = Document.objects.all()

    context = { 
                'Document_options':Document_options,

                 }
    return render(request, 'pages/search_data.html',context)


@login_required
def search_data(request):
    template_name = 'pages/search_data.html'

    #if request.method == 'GET':
        
    n = request.GET.get('n_comp')
    print(n)
    if n:
        global n_comp
        n_comp=n

    Document_id = request.GET.get('Document_id')
    Document_options = Document.objects.all()
    if Document_id:      
        doc = Document.objects.get(id=Document_id).document 
        global data
        data = pd.read_csv(doc,decimal=".",index_col=0)  
        global names
        names = data.index # ou data.index pour avoir les intitulés
        global features
        features = data.columns
        global dat 
        dat = data.to_html()

    args = {'data': dat ,'Document_options':Document_options,}
    return render(request , template_name,args)       
    


@login_required
def dataView(reqeust):
    template_name = 'pages/data.html'
    global dat 
    dat = data.to_html()
    args = {'data': dat ,}
    return render(reqeust , template_name,args)


@login_required
def data_Centrage_Réduction_View(reqeust):
    template_name = 'pages/Centrage_Réduction.html'

    global data
    print(data)
    print(data.shape)
    print(data.columns)
    print(data.index)
    
    # selection des colonnes à prendre en compte dans l'ACP
    nbc=len(data.index)
    global n_clust
    n_clust=nbc
    
    data_pca=data.iloc[:,0:nbc]
    print(data_pca)
    #data_pca = data[["Math","Phys","Franc","Scnt","Angl","Hist"]]
    #print(data_pca)

    # préparation des données pour l'ACP
    data_pca = data_pca.fillna(data_pca.mean()) # Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
    X = data_pca.values

    # Centrage et Réduction
    std_scale = preprocessing.StandardScaler().fit(X)
    global X_scaled
    X_scaled = std_scale.transform(X)
    print(X_scaled)
    #data=X_scaled

    args = {'X_scaled': X_scaled ,}
    return render(reqeust , template_name,args)
      
    

@login_required
def clusterView(request):
    template_name = 'pages/cluster.html'

    nc = request.GET.get('n_clust')
    print(nc)
    global n_clust
    if nc:
        n_clust=nc
    

    n_clust=int(n_clust)
    # Clustering par K-means
    km = KMeans(n_clusters= n_clust)
    global X_scaled
    km.fit(X_scaled)
    print(km)

    # Récupération des clusters attribués à chaque individu
    global clusters
    clusters = km.labels_

    global centroids
    centroids = km.cluster_centers_
    print(centroids)

    X_scaled=centroids
    args = {'centroids': centroids ,}
    return render(request , template_name,args)

"""
def affichage_p(request):
    f = matplotlib.figure.Figure()
    global pca
    global X_scaled
    pca = decomposition.PCA(n_components=3).fit(X_scaled)
    X_projected = pca.transform(X_scaled)
    global clusters
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters.astype(np.float), cmap = 'jet', alpha=.2)
    plt.title("Projection des {} individus sur le 1e plan factoriel".format(X_projected.shape[0]))
    #plt.show(block=False)
    FigureCanvasAgg(f)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    del buf
    return response

def affichage_cluster(request):
    f = matplotlib.figure.Figure()
    plt.figure()
    global centroids
    global X_scaled
    centroids_projected = pca.transform(centroids)
    plt.scatter(centroids_projected[:,0],centroids_projected[:,1])
    plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
    #plt.show()
    FigureCanvasAgg(f)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    X_scaled=centroids
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    del buf
    return response    
"""
@login_required
def pcaView(reqeust):
    template_name = 'pages/pca.html'
    global n_comp
    n_comp=int(n_comp)
    global pca
    pca = decomposition.PCA(n_components=n_comp)
    global X_scaled
    pca.fit(X_scaled)
    print(pca)

    # Eboulis des valeurs propres
    # display_scree_plot(pca)

    # Cercle des corrélations
    # pcs = pca.components_
    # display_circles(pcs, n_comp,pca, [(0,1),(2,3),(4,5)], labels = np.array(features))

    # Projection des individus
    # X_projected = pca.transform(X_scaled)
    # display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(names))

    # plt.show()

    args = {'pca': pca ,}
    return render(reqeust , template_name,args)

@login_required
    # display_scree_plot(pca)
def display_scree_plot(request):
    f = matplotlib.figure.Figure()
    global pca
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    #plt.show(block=False)
    FigureCanvasAgg(f)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    del buf
    return response



@login_required
def display_circles(reqeust):
    # Cercle des corrélations
    # pcs = pca.components_
    # display_circles(pcs, n_comp,pca, [(0,1),(2,3),(4,5)], labels = np.array(features))
    
    f = matplotlib.figure.Figure()

    global pca
    pcs = pca.components_
    global n_comp 
    axis_ranks=[(0,1),(2,3),(4,5)]
    labels = np.array(features)
    label_rotation=0
    lims=None

    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes    
        if d2 < n_comp:
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            # plt.show(block=False)

    FigureCanvasAgg(f)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    del buf
    return response

@login_required
def display_factorial_planes(reqeust):
    f = matplotlib.figure.Figure()


    global pca
    pcs = pca.components_
    global n_comp 
    axis_ranks=[(0,1),(2,3),(4,5)]
    global names
    labels = np.array(names)
    label_rotation=0
    lims=None
    global X_scaled
    X_projected = pca.transform(X_scaled)
    illustrative_var=None
    alpha=1

    for d1,d2 in axis_ranks:
        if d2 < n_comp:   

            # initialisation de la figure 
            fig = plt.figure(figsize=(7,6))
                
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            #plt.show(block=False)

    FigureCanvasAgg(f)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(f)
    del pca
    del X_scaled
    del X_projected

    global donnees
    donnees= None #Document.objects.get(description='data').document
    global data
    data = None #pd.read_csv(donnees,decimal=".",index_col=0) 

    response = HttpResponse(buf.getvalue(), content_type='image/png')
    del buf
    return response
