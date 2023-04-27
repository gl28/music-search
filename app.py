from flask import Flask, render_template, request
from embeddings import get_albums_for_query
from spotify import get_spotify_metadata

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    albums = get_albums_for_query(query)
    
    for album in albums:
       # TODO: cache spotify metadata
       link, art = get_spotify_metadata(album)
       album["spotify_link"] = link
       album["cover_art_url"] = art

    return render_template('index.html', albums=albums)

if __name__ == '__main__':
    app.run(debug=True)
