import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.feature_extraction.text as ext
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random

def wordsAndArtikels():
    file = open('artikel.txt' , 'r')
    words = []
    artikels = []
    for e in file:
        e = e.lower()
        if(len(e) >= 3):
            artikels.append(e[0] + e[1] + e[2])
            words.append(e[4:len(e)-1])

    file.close()

    return words,artikels


def convertWordsToNumbers(words):
    numberWords = []
    for word in words:
        numberWord = []
        for i in range(0,30):
            if len(word) > i:
                 numberWord.append(ord(word[i]))
            else:
                numberWord.insert(0 , 0)
        numberWords.append(numberWord)

    return numberWords

def convertWordToNumber(word):
    numberWord = []
    for i in range(0,30):
        if len(word) > i:
            numberWord.append(ord(word[i]))
        else:
            numberWord.insert(0 , 0)
    return numberWord

def convertArticlesToNumbers(articles):
    numberArticle = []
    for article in articles:
        if article == 'das':
            numberArticle.append([0,0,1])
        elif article == 'der':
            numberArticle.append([1,0,0])
        else:
            numberArticle.append([0,1,0])
    return numberArticle


def createFeature(random_size = 0.1):

    words, artikels = wordsAndArtikels()
    words = convertWordsToNumbers(words)

    artikels = convertArticlesToNumbers(artikels)
    # print(artikels)
    together = [artikels, words]
    together = np.array(together)
    together = together.transpose([1,0])


    train_x = ([together[i][1] for i in range(len(together))])[:int(len(words)*(1-random_size))]



    train_y = (([together[i][0] for i in range(len(together))]))[:int(len(artikels)*(1-random_size))]
    test_x = (([together[i][1] for i in range(len(words))]))[int(len(words) * (1 - random_size)):]
    test_y = (([together[i][0] for i in range(len(words))]))[int(len(artikels) * (1 - random_size)):]

    return train_x, train_y, test_x , test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = wordsAndArtikels()

