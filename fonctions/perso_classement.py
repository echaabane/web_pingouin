import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns


def unique_dict(dataframe, colonne):
    """mets les valeurs uniques d'une colonne dans un dictionnaire 1:valeurs unique'
    from sklearn.preprocessing import LabelEncoder fait quasiment la meme chose
    """
    a = 0
    dico = {}
    list_unique = list(dataframe[colonne].unique())

    for valeur in list_unique:
        dico[valeur] = a
        a += 1
    return dico


def separation_data_train_test_target(data, colonne_test):
    """
    separe le dataset en 4 donnée et cible, un jeu de test et un jeu d entrainement
    arg
        data: dataframe à couper
        colonne_test : string nom de la colonne cible
    return:
        data_train : data frame entrainement
        target_data_train : cible entrainement
        data_test : data de test
        target_data_test : cible de test
    """
    data_train, data_test = train_test_split(data)
    target_data_train = data_train.pop(colonne_test)
    target_data_test = data_test.pop(colonne_test)

    return data_train, target_data_train, data_test, target_data_test


def surapprentissage(target_train, predict_train, target_test, predict_test):
    print("rapport sur données d'entrainement :\n\n")
    print(classification_report(target_train, predict_train))
    print("\n\nrapport sur données de test:\n\n")
    print(classification_report(target_test, predict_test))


def rapport_confusion(predict_test, target_test, ax):
    # fait le rapport avec les matrice de confusion et rapport sklearn
    # fig,ax=plt.subplots()

    ConfusionMatrixDisplay.from_predictions(predict_test, target_test, ax=ax)
    ax.set_ylabel("Valeurs exactes")
    ax.set_xlabel("Valeurs prédites")
    ax.set_title("matrice de confusion")

    return ax


def rapport_confusion_model(model, data_test, target_test, ax):
    # fig,ax=plt.subplots()

    ConfusionMatrixDisplay.from_estimator(model, data_test, target_test, ax=ax)
    ax.set_ylabel("Valeurs exactes")
    ax.set_xlabel("Valeurs prédites")
    ax.set_title("Matrice de confusion")

    # pour info documentation sur le classification report
    # https://kobia.fr/classification-metrics-f1-score/
    # https://kobia.fr/classification-metrics-precision-recall/

    print("rapport de classification jeu de test")
    predict_test_logreg = model.predict(data_test)
    print(classification_report(target_test, predict_test_logreg))

    return ax


def inertie_interclassse(X, nb_cluster_mini=1, nb_cluster_max=10):
    """
    Ceci est inspiré très fortement du cours openclassroom: Réalisez une analyse exploratoire de données
    https://openclassrooms.com/fr/courses/4525281-realisez-une-analyse-exploratoire-de-donnees

    Trace un graphe pour voir inertie interclasse pour voir determiner le nombre de cluster
    parameter :
        X : Dataframe
        nb_cluster_mini : nombre de cluster minimun pour le graphe
        nb_cluster_max : nombre max pour le graphe
    """

    # Une liste vide pour enregistrer les inerties :
    intertia_list = []

    # Notre liste de nombres de clusters :
    k_list = range(nb_cluster_mini, nb_cluster_max)

    # Pour chaque nombre de clusters :
    for k in k_list:
        # On instancie un k-means pour k clusters
        kmeans = KMeans(n_clusters=k)

        # On entraine
        kmeans.fit(X)

        # On enregistre l'inertie obtenue :
        intertia_list.append(kmeans.inertia_)

    # trace le graphe
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.set_ylabel("intertia")
    ax.set_xlabel("n_cluster")

    ax = plt.plot(k_list, intertia_list)


def determination_cluster(df, nb_cluster, col_cluster="cluster"):
    """
    recherche des clusters par la methode du Kmeans de sklearn et rajoute une colonne cluster dans le dataframe
    parameter:
    df : dataframe a categoriser
    nb_cluster : nombre de groupe a effectuer
    col_cluster : nom de colonne a donné pour la colonne des cluster Default : cluster
    """
    kmeans = KMeans(n_clusters=nb_cluster)
    kmeans.fit(df)
    df[col_cluster] = kmeans.labels_


def dentogramme(
        df, methode="ward", nb_branche=10, taille=(12, 8), nom_colonne_cluster="cluster"
):
    """
    Dessine un dentogramme et rajoute un colonne avec les cluster trouvés.
    SRC : https://openclassrooms.com/fr/courses/4525281-realisez-une-
    analyse-exploratoire-de-donnees/5177936-effectuez
    -une-classification-hierarchique
    parameter :
        df: dataframe à analyser
        methode : methode voulue default WARD
        nb_branche : nombre de branche à afficher
        nom_colonne_cluster : nom de la colonne default 'cluster'
        taille : taille du graphe en pouce

    """
    Z = linkage(df, method=methode)
    fig, ax = plt.subplots(1, 1, figsize=taille)

    _ = dendrogram(Z, p=nb_branche, truncate_mode="lastp", ax=ax)

    plt.title("Classification Hierarchique ")
    plt.xlabel(
        "nombre de point dans le noeud(ou l index du point s'il n'y pas de parenthèse)."
    )
    plt.ylabel("Distance.")

    # determination des clusters :

    df[nom_colonne_cluster] = fcluster(Z, nb_branche, criterion="maxclust")

    # on retourne le graphe si on veut l afficher différemment
    return ax


def kmeans_bool(data_train, target_train, data_test, target_test, ax):
    """
    retourne un model de kmeans de sklearn avec 2 cluster 0 faux 1 vrais (un booléen quoi)
    arg:
        data_train et test : dataframe des données
        target train et test : cible
        colonne_cible : colonne du dataframe avec les valeurs cibles
    return:
        model sklearns Kmeans
    """
    # creation du modele
    # on fait un Kmeans avec 2 clusters vrais faux
    model_KMeans = KMeans(n_clusters=2, n_init="auto")

    # entrainement
    model_KMeans.fit(data_train)

    # target_kmeans_test = np.asarray(target_logis_test.astype(int))
    predict_test_KMeans = model_KMeans.predict(data_test)
    predict_test_KMeans = predict_test_KMeans.astype(bool)

    # Si les clusters ne sont pas bien situé, je refait le Kmeans en changeant les centres initiaux
    kmeans_confusion = confusion_matrix(predict_test_KMeans, target_test)
    somme_defaut = kmeans_confusion[0, 0] + kmeans_confusion[1, 1]
    valeur_bonne = kmeans_confusion[0, 1] + kmeans_confusion[1, 0]
    center = np.empty(shape=(2, 6))
    if valeur_bonne < somme_defaut:
        center = model_KMeans.cluster_centers_
    else:
        center = np.array(
            [model_KMeans.cluster_centers_[1], model_KMeans.cluster_centers_[0]]
        )

    # recréation du modele avec les centres "bien placé"
    # pour eviter le warning du au placement des centres: n_init=1 permet d'eviter le warning
    model_KMeans = KMeans(n_clusters=2, init=center, n_init=1)

    # entrainement
    model_KMeans.fit(data_train)

    # analyse du modele
    predict_test_KMeans = model_KMeans.predict(data_test)
    predict_test_KMeans = predict_test_KMeans.astype(bool)

    predict_test = model_KMeans.predict(data_test)
    predict_test = predict_test.astype(bool)

    predict_train = model_KMeans.predict(data_train)
    predict_train = predict_train.astype(bool)

    surapprentissage(
        target_test=target_test,
        target_train=target_train,
        predict_test=predict_test,
        predict_train=predict_train,
    )

    ax = rapport_confusion(
        predict_test=predict_test_KMeans, target_test=target_test, ax=ax
    )

    return model_KMeans, ax


def KNNtest(
        data_train, target_train, data_test, target_test, ax, ax_knn, kmin=1, kmax=10
):
    """renvoie un modele KNN
    arg: data , dataframe a analyser
        colonne_cible : nom de la colonnne cible
    returns:
        courbe apprentissage : dataframe avec les scores train et test

    """

    courbe_apprentissage = pd.DataFrame(columns=["score_train", "score_test"])
    for a in range(kmin, kmax):
        modelvoisin = KNeighborsClassifier(n_neighbors=a)
        modelvoisin.fit(data_train, target_train)
        score_train = modelvoisin.score(data_train, target_train)
        score_test = modelvoisin.score(data_test, target_test)
        liste_score = [score_train, score_test]
        courbe_apprentissage.loc[a] = liste_score

    # fig,ax_knn = plt.subplots()

    sns.lineplot(courbe_apprentissage, ax=ax_knn)
    ax_knn.set_title("score du KNN en fonction du nombre de voisins")
    ax_knn.set_xlabel("nombre de voisins")
    ax_knn.set_ylabel("score")

    courbe_apprentissage["ecart"] = (
            courbe_apprentissage["score_train"].abs()
            - courbe_apprentissage["score_test"].abs()
    )
    nb_voisin = courbe_apprentissage["ecart"].idxmin()
    print("Le nombre de voisins choisi automatiquement est de :", nb_voisin)

    model_voisin = KNeighborsClassifier(n_neighbors=nb_voisin)
    model_voisin.fit(data_train, target_train)

    predict_test_KNN = model_voisin.predict(data_test)
    predict_test_KNN = predict_test_KNN.astype(bool)

    predict_train_KNN = model_voisin.predict(data_train)
    predict_train_KNN = predict_train_KNN.astype(bool)

    surapprentissage(
        target_test=target_test,
        target_train=target_train,
        predict_test=predict_test_KNN,
        predict_train=predict_train_KNN,
    )
    # affichage matrice de confusion
    ax = rapport_confusion(
        predict_test=predict_test_KNN, target_test=target_test, ax=ax
    )

    return model_voisin, ax, ax_knn


def regression_logistique(
        data_train, target_train, data_test, target_test, ax, max_iter=100
):
    """
    renvoie un modele de regression logistique sklearn
    arg:
        data : dataframe dsiponible en apprentissage

    return:
        modele sklearn de regression logistique
    """

    # création du model (par defaut)
    model_reg_logis = LogisticRegression(max_iter=max_iter)

    # entrainement
    model_reg_logis.fit(data_train, target_train)

    predict_train = model_reg_logis.predict(data_train)
    predict_test = model_reg_logis.predict(data_test)

    surapprentissage(
        target_test=target_test,
        target_train=target_train,
        predict_test=predict_test,
        predict_train=predict_train,
    )
    # affichage matrice de confusion
    ax = rapport_confusion(predict_test=predict_test, target_test=target_test, ax=ax)

    return model_reg_logis, ax


def naive_bayes_gaussian(data_train, target_train, data_test, target_test, ax):
    """renvoie une objet sklearn de Naive_bayes
    Arg :
        data : data

    return
        gnb : objet sklearn de classification
    """

    # creation de l'objet sklearn
    gnb = GaussianNB()
    gnb.fit(data_train, target_train)

    # scoring

    predict_train = gnb.predict(data_train)
    predict_test = gnb.predict(data_test)

    surapprentissage(
        target_test=target_test,
        target_train=target_train,
        predict_test=predict_test,
        predict_train=predict_train,
    )
    # affichage matrice de confusion
    ax = rapport_confusion(predict_test=predict_test, target_test=target_test, ax=ax)

    return gnb, ax


def SVM(data_train, target_train, data_test, target_test, ax):
    """renvoie une objet sklearn de SVM
    Arg :
        data : data

    return
        gnb : objet sklearn de classification
    """

    # creation de l'objet sklearn
    svm = SVC()
    svm.fit(data_train, target_train)

    # scoring

    predict_train = svm.predict(data_train)
    predict_test = svm.predict(data_test)

    surapprentissage(
        target_test=target_test,
        target_train=target_train,
        predict_test=predict_test,
        predict_train=predict_train,
    )
    # affichage matrice de confusion
    ax = rapport_confusion(predict_test=predict_test, target_test=target_test, ax=ax)

    return svm, ax
