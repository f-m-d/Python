import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#Fetch data and format it
data = fetch_movielens(min_rating=4.0)

#Printing training data and test data
print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss='warp')

#Traina il modello con i dati train, con 30 epochs a 2 thread
model.fit(data['train'], epochs=30, num_threads=2)

def simple_recommendation(model, data, user_ids):
    
    #Number of user and items
    n_users, n_items = data['train'].shape

    for user_id in user_ids:
        #Movie the already like
        known_positives=data['item_labels'][data['train'].tocsr()[user_id].indices]

        #Movies our model predict they will like
        scores = model.predict(user_id, np.arange(n_items))

        #Order them by the most liked
        top_items=data['item_labels'][np.argsort(-scores)]

        #Print out all the results
        print("User %s" % user_id)
        print("     Knowkn positives:")

        #Stampa fino a 3 elementi dei conosciuti
        for x in known_positives[:3]:
            print("     %s" % x)

        print("     Recomended:")

        #Stampa fino a tre elementi dei previsti
        for x in top_items[:3]:
            print("     %s" %  x)

#Chiamo il metodo con valori
simple_recommendation(model, data, [3, 25, 450])

#Advanced: Write a new method
#for fetching and formatting a new dataset
#Train it on 3 different models
#Print results from the best one

#Utilizzo il dataset di StackExchange
from lightfm.datasets import fetch_stackexchange

#Notice me about changing dataset
print("##########################################")
print("##### STACK EXCHANGE: CROSSVALIDATED #####")
print("##########################################")
print("")

#data_stackexchange = fetch_stackexchange(dataset='crossvalidated')

#print(repr(data_stackexchange['train']))
#print("")
#print(repr(data_stackexchange['test']))


#Traina 3 diversi tipi di model