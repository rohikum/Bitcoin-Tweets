import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))	
		conf = 1.0 * choice_votes / len(votes)
		return conf


short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

all_words = []
documents = []

# J is adject, R is adverb, and V is verb
allowed_word_types = ['J', 'R', 'V']
#allowed_word_types = ['J']

for p in short_pos.split('\n'):
	documents.append( (p, 'pos') )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())


for p in short_neg.split('\n'):
	documents.append( (p, 'neg') )
	words = word_tokenize(p)
	pos = nltk.pos_tag(words)
	for w in pos:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

save_documents = open('documents.pickle', 'wb')
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_word_features = open('word_features5k.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Accuracy: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Accuracy: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("MNBClassifier5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("BernoulliNB.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("LinearSVC_classifier5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier,
									LinearSVC_classifier, LogisticRegression_classifier)
print("voted_classifier Accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
	feats = find_features(text)
	return [voted_classifier.classify(feats), voted_classifier.confidence(feats)]

print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons..."))
print(sentiment("This movie was utter junk! There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
