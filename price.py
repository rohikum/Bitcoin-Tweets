import requests
import pandas as pandas
import datetime
import sqlite3
import json
import time

while True:
	url = 'https://api.cryptowat.ch/markets/coinbase-pro/btcusd/ohlc'
	r = requests.get(url)
	temp = r.json()['result']['60']

	for i in range(len(temp)):

		conn = sqlite3.connect('MasterDB.db')
		c= conn.cursor()
		del temp[i][-2]
		temp[i][0] = datetime.datetime.fromtimestamp(temp[i][0]).strftime('%Y-%m-%d %H:%M:%S')
		
		try:
			c.execute('CREATE TABLE masterData(time_Stamp INTEGER, openVal REAL, highVal REAL, lowVal REAL, closeVal REAL, volumeVal REAL)')
			c.execute('INSERT INTO masterData VALUES (?, ?, ?, ?, ?, ?)', tuple(temp[i]))
			conn.commit()

		except:
			try:
				c.execute('SELECT * FROM masterData ORDER BY time_Stamp DESC LIMIT 1')
				check_me = c.fetchall()
				if check_me[0][0] < temp[i][0]:
					c.execute('INSERT INTO masterData VALUES (?, ?, ?, ?, ?, ?)', tuple(temp[i]))
					conn.commit()
			except:
				check_me = 0
	conn.close()

	print('Done!! Sleeping for 60 seconds')
	time.sleep(60)
	print('Back to work!')