import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

textArray = []

def readAndProcess(fileName):
    with open(fileName, 'r', encoding="utf8") as file:
        for line in file:
            textArray.append(line.strip().split('|')[2])

def bagOfWordsCreator(textArray):
        vectorizer = CountVectorizer()
        vectorizer.fit(textArray)
        BOW = vectorizer.transform(textArray)     
        """
        with open("BagOfWords.txt", "w") as file:
            for i in BOW:
                file.write(str(i)) 
        """
        return BOW
    
def plotClusters(simplifiedCLusters, labels, title):
    plt.scatter(simplifiedCLusters[:, 0], simplifiedCLusters[:, 1], c=labels,s=1)
    plt.title(title)
    plt.show()
    
def cosineDistance(bagOfWords):
    cosineSimilarities = linear_kernel(bagOfWords, bagOfWords)
    cosineSimilarities = cosineSimilarities.flatten()
    cosineSimilarities = cosineSimilarities[cosineSimilarities != 1]
    """
    with open("CosineDistances.txt", "w") as file:
            for i in cosineSimilarities:
                file.write(str(i))
    """
    plt.hist(cosineSimilarities, bins = 500)
    plt.show()
    
def euclideanDistance(bagOfWords):
    euclideanSimiliarties = euclidean_distances(bagOfWords)
    euclideanSimiliarties = euclideanSimiliarties.flatten()
    euclideanSimiliarties = euclideanSimiliarties[euclideanSimiliarties != 0]
    """
    with open("EuclideanDistances.txt", "w") as file:
        for i in euclideanSimiliarties:
            file.write(str(i))    
    """
    plt.hist(euclideanSimiliarties, bins = 500)
    plt.show()

def dbscanCosine(bagOfWords):
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    dbscan.fit(bagOfWords)
    labels = dbscan.labels_
    print(labels)
    silScore = silhouette_score(bagOfWords, labels)
    print(f'Silhouette Score for Cosine DBSCAN: {silScore}')
    simplifiedCLusters = PCA(n_components=2).fit_transform(bagOfWords.toarray())
    plotClusters(simplifiedCLusters,labels, 'DBSCAN Cosine Clusters')
    
def aggCosine(bagOfWords):
    agglo = AgglomerativeClustering(n_clusters=10, metric='cosine', linkage='average')
    agglo.fit(bagOfWords.toarray())
    labels = agglo.labels_
    print(labels)
    silScore = silhouette_score(bagOfWords, labels)
    print(f'Silhouette Score for Cosine Agglomerative Hierarchical: {silScore}')
    simplifiedCLusters = PCA(n_components=2).fit_transform(bagOfWords.toarray())
    plotClusters(simplifiedCLusters,labels, 'Agglomerative Hierarchical Cosine Clusters')

def aggEuclidean(bagOfWords):
    agglo = AgglomerativeClustering(n_clusters=10, metric='euclidean', linkage='average')
    agglo.fit(bagOfWords.toarray())
    labels = agglo.labels_
    print(labels)
    silScore = silhouette_score(bagOfWords, labels)
    print(f'Silhouette Score for Euclidean Agglomerative Hierarchical: {silScore}')
    simplifiedCLusters = PCA(n_components=2).fit_transform(bagOfWords.toarray())
    plotClusters(simplifiedCLusters,labels, 'Agglomerative Hierarchical Euclidean Clusters')
    
    
def kMeans(bagOfWords):
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(bagOfWords)
    labels = kmeans.labels_
    print(labels)
    silScore = silhouette_score(bagOfWords, labels)
    print(f'Silhouette Score for Kmeans: {silScore}')
    simplifiedCLusters = PCA(n_components=2).fit_transform(bagOfWords.toarray())
    plotClusters(simplifiedCLusters,labels, 'Kmeans Clusters')
    
readAndProcess("cnnhealth.txt")
bagOfWords = bagOfWordsCreator(textArray)

aggEuclidean(bagOfWords)