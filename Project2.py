import re
import nltk
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

textArray = []

def readAndProcess(fileName):
    #makes the list of stopwords
    nltk.download('stopwords') #only needs to be downloaded once
    words=stopwords.words('english')
    #adding custom stopwords
    addOnWords = ['a','to','cnn','rt','of','it','at']
    for i in addOnWords:
        words.append(i)
    stemmer = SnowballStemmer("english")
    stopWords= set(words)
    
    with open(fileName, 'r', encoding="utf8") as file:
        for line in file:
            tempString = ""
            temp = line.strip().split('|')[2]
            tempArray = temp.split(' ')
            for i in tempArray:
                if i[:4] == 'http':
                    i = ' '
                
                tempString += i+ ' '
            #removes digits and making everything lowercase
            tempString = re.sub(r'\d+','', tempString).lower()
            #removes special characters
            tempString = tempString.replace('\'',"")
            tempString = re.sub(r'\W+',' ', tempString)
            #removes stop words
            tempString=' '.join([word for word in tempString.split() if word not in stopWords])
            #print(tempString)
            textArray.append(tempString)      
            
def bagOfWordsCreator(textArray):
        #Create Bag of Words
        vectorizer = CountVectorizer()
        vectorizer.fit(textArray)
        BOW = vectorizer.transform(textArray)
        #Generate Statistics
        wordCount = BOW.toarray().sum(axis=0)
        wordFreq = [(word, wordCount[idx]) for word, idx in vectorizer.vocabulary_.items()]
        wordFreq = sorted(wordFreq, key=lambda x: x[1], reverse=True) 
        #Print Statistics to file
        with open("Feature Matrix Statistics.txt", "w") as file:
            file.write("Number of Documents: " + str(len(textArray)) + '\n')  
            file.write("Number of Tokens: " + str(sum(wordCount)) + '\n')  
            file.write("Number of Unique Terms: " + str(len(wordFreq)) + '\n')    
            file.write("Average Terms: " + str(sum(wordCount)/len(textArray)) + '\n')    
            for i in wordFreq:
                file.write(str(i) + '\n')    
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
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(bagOfWords)
    labels = kmeans.labels_
    print(len(labels))
    silScore = silhouette_score(bagOfWords, labels)
    print(f'Silhouette Score for Kmeans: {silScore}')
    simplifiedCLusters = PCA(n_components=2).fit_transform(bagOfWords.toarray())
    plotClusters(simplifiedCLusters,labels, 'Kmeans Clusters')
    
readAndProcess("cnnhealth.txt")
bagOfWords = bagOfWordsCreator(textArray)
