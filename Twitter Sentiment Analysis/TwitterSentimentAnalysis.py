import tweepy
import csv

from textblob import TextBlob

#Chiave dell'API pubblica e privata
consumerKey = 
consumerSecret = 

#Token pubblico e privato per l'accesso
accessToken = 
accessTokenSecret = 

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

#Abbiamo accesso all'API di Twitter
api = tweepy.API(auth)


#Ricerca di tweets per una determinata parola
#Attenzione al char encoding per lingue con caratteri speciali
word='Italia'
public_tweets = api.search(word)

#Mostra i tweets e la loro analisi sentimentale: polarity e subjectivity
for tweet in public_tweets:
    print(tweet.text)
    analysis=TextBlob(tweet.text)
    print(analysis.sentiment)

#Parte aggiuntiva: creare un file CSV 

#Apro o creo un file con nome default + parola cercata
twitter_file_name='Twitter_Analysis_On_' + word + '.csv'
twitter_file=open(twitter_file_name, 'w+')

#Definisco il writer e creo l'header
twitter_writer = csv.writer(twitter_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
twitter_writer.writerow(['Tweet', 'Analysis'])
twitter_writer.writerow(['', ''])

for tweet in public_tweets:
    analysis=TextBlob(tweet.text)
    twitter_writer.writerow([tweet.text, analysis.sentiment])
    twitter_writer.writerow(['', ''])