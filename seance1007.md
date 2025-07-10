-> Analyse de plusieurs capteurs à l'oscillo pour voir si on peut

Voir photos pour montrer qu'on peut détecter correctement temporellement la différence entre les différents capteurs

-> 20 microseconds de temps pour pouvoir voir correctement la différence entre 2 signaux avec les distances qu'on a ici
-> F = 1/T donc 1/(20*10**-6) càd 50kHz de freq d'échantillonnage

-> il faut utiliser double buffers circulaires + carte sd pour arriver à cette fréquence car sinon limité à 11KHz

""" marche bof bof
Calcul théorique de freq en USB:
- baud rate = n bits par seconde = 1 000 000
- on envoie environ 25 * 8 bits (entre 2 et 3 par capteurs + les virgules), sachant que un caractère = 8 bits donc 200 bits par datapoint
- donc on a en théorie 1 000 000 / 200 = 5000
"""


On a besoin de booster la freq de sampling donc utilisation d'un double buffer circulaire


### taille des buffers

sachant qu'on a une freq d'échantillonnage de chaque capteur de 50kHz donc -> 50 000 points/s
-> Si on a un buffer de 20ms par exemple, on a 50 000 / 500 * 5; * 5 car 5 capteurs /500 car on a que 20ms (1000/20 -> 500 points/ms)


-> faire les kicad des différents montages testés pour les comparer et montrer les problèmes etc


### tension
Tension mesurée à l'oscillo = ordre de 150V donc les estimations avec pont diviseur sont pas trop correctes car le pont diviseur de tension absorbe une partie de la charge