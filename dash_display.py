import dash
from dash.dependencies import Event, Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3
import pandas as pd

conn = sqlite3.connect('MasterDB.db', check_same_thread=False)

#conn = sqlite3.connect('twitter.db')
#c = conn.cursor()

sentiment_colors = {-1:"#EE6055",
                    -0.5:"#FDE74C",
                     0:"#FFE6AC",
                     0.5:"#D0F2DF",
                     1:"#9CEC5B",}


app_colors = {
    'background': '#0C0F0A',
    'text': '#FFFFFF',
    'sentiment-plot':'#41EAD4',
    'volume-bar':'#FBFC74',
    'someothercolor':'#FF206E',
}

POS_NEG_NEUT = 0.1

MAX_DF_LENGTH = 100

app = dash.Dash(__name__)
app.layout = html.Div([
		
        dcc.Checklist(
            id = 'clts',
            values=[],
            options=[],
            labelStyle={'display': 'inline-block'},
            style={"height" : "0.8vh", "width" : "90vw"}
        ),  

        html.Div(dcc.Graph(id='price-graph', animate=False, 
        	style = {
        	'height': '55vh',
        	'width': '130vh',
        	'float': 'left',
        	'display': 'inline-block'})),

        html.Div(dcc.Graph(id='live-graph', animate=False,
        	 style = {
        	'height': '45vh',
        	'width': '130vh',
        	'float': 'left',
        	'display': 'inline-block'})),


        html.Div(dcc.Graph(id='sentiment-pie', animate=False,
        	 style = {
        	'height': '50vh',
        	'width': '85vh',
        	'float': 'top-right',
        	'display': 'inline-block'})), 

        html.Div(id="recent-tweets-table", 
        	 style = {
        	'height': '50vh',
        	'width': '85vh',
        	'float': 'top',
        	'display': 'inline-block'}), 
        
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),

        dcc.Interval(
            id='recent-table-update',
            interval=2*1000
        ),

        dcc.Interval(
            id='price-update',
            interval=1*1000
        ),

        dcc.Interval(
            id='sentiment-pie-update',
            interval=1*1000
        ),

    ], style={'backgroundColor': app_colors['background'], 'margin-top':'0px', 'height':'850px',},
)

def df_resample_sizes(maxlen=MAX_DF_LENGTH):
	df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 5000", conn)
	df.sort_values('unix', inplace = True)
	df['unix'] = df['unix'].apply(lambda x: x - 18000000)
	df['date'] = pd.to_datetime(df['unix'], unit = 'ms')
	df.set_index('date', inplace = True)
	init_length = len(df)
	df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
	df_len = len(df)
	resample_amt = 100
	vol_df = df.copy()
	vol_df['volume'] = 1
	ms_span = (df.index[-1] - df.index[0]).seconds * 1000
	rs = int(ms_span / maxlen)
	df = df.resample('{}ms'.format(int(rs))).mean()
	df.dropna(inplace=True)
	vol_df = vol_df.resample('{}ms'.format(int(rs))).sum()
	vol_df.dropna(inplace=True)
	df = df.join(vol_df['volume'])
	global tweet_start_time
	tweet_start_time = min(df.index)
	return df

def quick_color(s):
    # except return bg as app_colors['background']
    if s >= POS_NEG_NEUT:
        # positive
        return "#002C0D"
    elif s <= -POS_NEG_NEUT:
        # negative:
        return "#270000"

    else:
        return app_colors['background']

def generate_table(df, max_rows=5):
    return html.Table(className="responsive-table",
                      children=[
                          html.Thead(
                              html.Tr(
                                  children=[
                                      html.Th(col.title()) for col in df.columns.values],
                                  style={'color':app_colors['text']}
                                  )
                              ),
                          html.Tbody(
                              [
                                  
                              html.Tr(
                                  children=[
                                      html.Td(data) for data in d
                                      ], style={'color':app_colors['text'],
                                                'background-color':quick_color(d[2])}
                                  )
                               for d in df.values.tolist()])
                          ]
    )



@app.callback(Output('price-graph', 'figure'),
			events = [Event('price-update', 'interval')])
def update_price_ohlc():
	try:
		df1 = pd.read_sql("SELECT * FROM masterData ORDER BY time_Stamp DESC LIMIT 500", conn)

		df1.sort_values('time_Stamp', inplace = True)
		df1.set_index('time_Stamp', inplace = True)

		X = df1.index
		data = go.Ohlc(
			x = X,
			open = df1.openVal,
			high = df1.highVal,
			low = df1.lowVal,
			close = df1.closeVal,
			)
	
		return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)], rangeslider = dict(visible = False), showgrid = True, gridcolor = '#313233'),
                                                      yaxis=dict(range=[min(df1.lowVal),max(df1.highVal)], title='Price (USD)', side='left', showgrid = True, gridcolor = '#313233'),
                                                      title='Live Bitcoin Price',
                                                      font={'color':app_colors['text']},
                                                      plot_bgcolor = app_colors['background'],
                                                      paper_bgcolor = app_colors['background'],
                                                      showlegend=False,)}
	except Exception as e:
		with open('errors.txt', 'a') as f:
			f.write(str(e))
			f.write('\n')



@app.callback(Output('live-graph', 'figure'),
			events = [Event('graph-update', 'interval')])
def update_graph_scatter():
	try:
		df = df_resample_sizes()
		#df = df.resample('10s').mean()
		X = df.index
		Y = df.sentiment_smoothed.values
		Y2 = df.volume.values
		data = go.Scatter(
			x = X,
			y = Y,
			name = 'Scatter',
			mode = 'lines',
			yaxis = 'y2',
			line = dict(color = 'aqua',
                            width = 1.5,)
			)
		data2 = go.Bar(
                x=X,
                y=Y2,
                name='Volume',
                marker=dict(color='orange'),
                )

		return {'data': [data,data2],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)], showgrid = True, gridcolor = '#313233'),
                                                          yaxis=dict(range=[min(Y2),max(Y2*4)], title='Volume', side='right', showgrid = True),
                                                          yaxis2=dict(range=[min(Y),max(Y)], side='left', overlaying='y',title='Sentiment', showgrid = True, gridcolor = '#313233'),
                                                          title='Live Twitter sentiment for Bitcoin',
                                                          font={'color':app_colors['text']},
                                                          plot_bgcolor = app_colors['background'],
                                                          paper_bgcolor = app_colors['background'],
                                                          showlegend=False)}

	except Exception as e:
		with open('errors.txt', 'a') as f:
			f.write(str(e))
			f.write('\n')


@app.callback(Output('recent-tweets-table', 'children'),
              events=[Event('recent-table-update', 'interval')])        
def update_recent_tweets(): 	
	df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 5", conn)
	df['unix'] = df['unix'].apply(lambda x: x - 18000000)
	df['date'] = pd.to_datetime(df['unix'], unit='ms')
	df = df.drop(['unix'], axis=1)
	df = df[['date', 'tweet','sentiment']]
	return generate_table(df, max_rows=5)


@app.callback(Output('sentiment-pie', 'figure'),
              events=[Event('sentiment-pie-update', 'interval')])
def update_pie_chart():
	sentiment_pie_dict = []
	temp_pos = 0
	temp_neg = 0
	temp_neu = 0
	df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 5000", conn)
	df['unix'] = df['unix'].apply(lambda x: x - 18000000)
	df['date'] = pd.to_datetime(df['unix'], unit='ms')
	df = df.drop(['unix'], axis=1)
	df = df[['date', 'tweet','sentiment']]
	sentiment_values = list(df['sentiment'])
	for i in range(len(sentiment_values)):
		if sentiment_values[i] > 0:
			temp_pos += 1
		if sentiment_values[i] < 0:
			temp_neg += 1
		if sentiment_values[i] == 0:
			temp_neu += 1
	sentiment_pie_dict.append(temp_pos)
	sentiment_pie_dict.append(temp_neg)
	sentiment_pie_dict.append(temp_neu)
	labels = ['Positive','Negative', 'Neutral']
	try: pos = sentiment_pie_dict[0]
	except: pos = 0
	try: neg = sentiment_pie_dict[1]
	except: neg = 0
	try: neu = sentiment_pie_dict[2]
	except: neu = 0
	values = [pos, neg, neu]
	colors = ['#007F25', '#800000', '#b3b8bf']
	trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value', 
                   textfont=dict(size=20, color=app_colors['text']),
                   marker=dict(colors=colors, 
                               line=dict(color=app_colors['background'], width=2)))

	return {"data":[trace],'layout' : go.Layout(
                                                  title='Sentiment Distribution',
                                                  font={'color':app_colors['text']},
                                                  plot_bgcolor = app_colors['background'],
                                                  paper_bgcolor = app_colors['background'],
                                                  showlegend=True)}


if __name__ == '__main__':
	app.run_server(debug = True)
