import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_metadata(album):
    query = album["title"] + " :" + album["artist"]
    results = sp.search(q=query, type='album', limit=1)

    if results['albums']['total'] > 0:
        album = results['albums']['items'][0]
        print(album)
        album_link = album['external_urls']['spotify']
        cover_art_link = album['images'][0]['url']
        return album_link, cover_art_link
    else:
        return None, None

if __name__ == "__main__":
    query = input("Enter a query to search for albums: ")
    print(get_spotify_metadata(query))