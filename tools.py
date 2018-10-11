import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model

def k_cross_validation_sets(k,x,y):
    cross_validation_x=[]
    cross_validation_y=[]
    size=y.shape[0]
    k_size=int(size/k)
    index=np.random.permutation(range(size))
    for i in range(1,k+1):
        temp_x=x[index[(i-1)*k_size+1:i*k_size],:]
        temp_y=y[index[(i-1)*k_size+1:i*k_size],:]
        cross_validation_x.append(temp_x)
        cross_validation_y.append(temp_y)

    return cross_validation_x, cross_validation_y

def test_sets(k,i):
    return [j-1 for j in range(1,k+1) if i!=j]

def create_sets(training_ratio,x,y):
    size=y.shape[0]
    size_test_set=int(size*(1-training_ratio))
    index=np.random.permutation(range(size))
    test_set=x[index[:size_test_set],:]
    training_set=x[index[size_test_set:size],:]
    y_test=y[index[:size_test_set]]
    y_training=y[index[size_test_set:size]]

    return test_set, training_set, y_test, y_training

def display_tab(name, tab):
    print(name)
    for item in tab:
        print( str(item) )

def display_graphs(tab):
    defaults={'color':'','label':'','xlabel':'','ylabel':'','title':''}
    plt.figure(figsize=(20,10))
    i=1
    for item in tab:
        defaults.update(item)
        plt.subplot(int(np.sqrt(len(tab)))+1,int(np.sqrt(len(tab)))+1,i)
        plt.scatter(defaults['x'],defaults['y'],c=defaults['color'],label=defaults['label'])
        plt.xlabel(defaults['xlabel'])
        plt.ylabel(defaults['ylabel'])
        plt.legend()
        plt.title(defaults['title'])
        i+=1
def display_graph(tab):
    defaults={'color':'','label':'','xlabel':'','ylabel':'','title':''}
    plt.figure()
    for item in tab:
        defaults.update(item)
        plt.scatter(defaults['x'],defaults['y'],c=defaults['color'],label=defaults['label'])
        plt.xlabel(defaults['xlabel'])
        plt.ylabel(defaults['ylabel'])
        plt.title(defaults['title'])
        plt.legend()
    plt.show()

def gen_linear(a,b,eps,nbex):
    #Generate points in d-dimension  Y= a*X +b + epsilon such a follows a -5,5 uniform law, b the offset and epsilon
    # a gaussian noise with variance=eps

    #differentiate cases where a is a tab and where a is a int
    if isinstance(a, int):
        size=1
    else:
        a=np.array(a)
        size=a.shape[0]
        a=a.reshape(-1,1)

    x=np.random.uniform(-5,5,(nbex,size))
    epsilon= np.random.normal(0,eps,nbex)
    epsilon=epsilon.reshape(-1,1)
    b=b*np.ones((nbex,1));
    y=x.dot(a) + b +epsilon

    #check the size of the output
    assert(x.shape==(nbex,size) )
    assert(y.shape==(nbex,1))

    return x.reshape(-1,size),y.reshape(-1,1)

def mse(yhat,y):
    size= y.shape[0]

    assert(y.shape==yhat.shape)

    return ((((y-yhat).T).dot(y-yhat))*1./size)[0][0]

def plot_data(data,labels=None):
    """
    Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    cols,marks = ["red", "blue","green", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data, f, step=20, alpha_c=1):
    """ Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),256, alpha=alpha_c)

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """ Generateur de donnees,
        :param centerx: centre des gaussiennes
        :param centery:
        :param sigma: des gaussiennes
        :param nbex: nombre d'exemples
        :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
        :param epsilon: bruit dans les donnees
        :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//2))
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex//2))
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),int(nbex//4)),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),int(nbex/4))))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(int(nbex//2))))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1

    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation((range(y.size)))
    data=data[idx,:]
    y=y[idx]
    return data,y
