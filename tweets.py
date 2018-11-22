from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sqlite3
from unidecode import unidecode
from textblob import TextBlob
import requests

conn = sqlite3.connect('MasterDB.db')
c = conn.cursor()

def create_table():
	c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
	conn.commit()

create_table()

ckey = '******************************'   #add your ckey from twitter
csecret = '*************************************'   #add your csecret from twitter
atoken = '*********************************'   #add your atoken from twitter   
asecret = '******************************'   #add your asecret from twitter

class listener(StreamListener):

	def on_data(self, data):
		try:
			data = json.loads(data)
			tweet = unidecode(data['text'])
			time_ms = data['timestamp_ms']
			
			analysis = TextBlob(tweet)
			sentiment = round(analysis.sentiment.polarity,2)

			print(time_ms, tweet, sentiment)
			c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)", 
						(time_ms, tweet, sentiment))
			conn.commit()

		except KeyError as e:
			print(str(e))
		return(True)

	def on_error(self, status):
		print (status)

while True:
	try:
		auth = OAuthHandler(ckey, csecret)
		auth.set_access_token(atoken, asecret)
		twitterStream = Stream(auth, listener())
		twitterStream.filter(track=['bitcoin'])
	except Exception as e:
		print(str(e))
		time.sleep(5)