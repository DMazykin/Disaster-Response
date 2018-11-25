import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly import tools
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_with_cat', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    genre_counts = df.groupby('genre').count()['message'].sort_values()
    genre_names = list(genre_counts.index)

    
    labels = df.columns[4:].tolist()
    messages_per_category = df[labels].sum().sort_values().tail(10)
    categories = list(messages_per_category.index.str.replace("_", " "))

    dataset = {
        'n_records': df.shape[0],
        'n_categories': len(labels)}

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    # Plot 1 with two subplots:

    fig1 = tools.make_subplots(rows=1, cols=2, print_grid=False,
                                subplot_titles=('Total number of messages by genre',
                                                'Number of categories per message'))
    # Trace 1: Number of categories per message

    for i in range(0, len(genre_names)):
        trace1 = dict(
            type='box',
            x = df[df['genre']==genre_names[i]][labels].sum(axis=1).values,
            name = genre_names[i])
        
        fig1.append_trace(trace1, 1, 2)
 
    # Trace 2: Total number of messages by genre

    trace2 = dict(type='bar',
                y=genre_names,
                x=genre_counts,
                orientation='h',
                hoverinfo='none',
                showlegend=False,
                marker=dict(color=plotly.colors.DEFAULT_PLOTLY_COLORS[:len(genre_names)]))

    fig1.append_trace(trace2, 1, 1)

    # Add annotation for Trace 2
    
    annotations = ()
    for xi, yi in zip(genre_counts, genre_names):
        annotations = annotations + (dict(x=xi,
                            y=yi,
                            xref='x1', 
                            yref='y1',
                            text=str(xi),
                            xanchor='left',
                            showarrow=False,
                            yanchor='middle'),)

    fig1.layout.annotations = fig1.layout.annotations + annotations
    
     # Plot 2
        
    fig2 = {'data': [dict(type='bar',
                    y=categories,
                    x=messages_per_category/len(df),
                    orientation = 'h',
                    hoverinfo='none'),

                dict(type='bar',
                    y=categories,
                    x=(len(df)-messages_per_category)/len(df),
                    marker=dict(color='rgb(204,204,204)'),
                    hoverinfo='none',
                    orientation = 'h')],
            'layout': dict(
                barmode='stack',
                title="Top 10 most frequent categories",
                xaxis=dict(tickformat=".2%", 
                            domain=[0.1, 0.9],
                            title="Apearance of category in Training Dataset"),
                margin=dict(pad=5),
                showlegend=False,
                annotations=[dict(
                        x=xi+0,
                        y=yi,
                        text=str("{:.2%}".format(xi)),
                        xanchor='left',
                        yanchor='middle',
                        showarrow=False) for yi, xi in zip(categories, messages_per_category/len(df))])
            }
    
    graphs = [fig1, fig2]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, dataset=dataset)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()