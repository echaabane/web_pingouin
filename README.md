Bonjour,

    Information importante : un fichier .env est fournie avec une clé secrete = "secret". Ne l utiliser pas 
    en production. 

A la racine : des fichiers notebook avec des Analyses effectuées sur différents supports.
Pour l'instant il n'y a que le Passion_Pingouins que j'avais crée pour expliquer les IA à mes filles.

Ainsi que les fichiers pour Flask run.py pour lancer une application en local et config.py avec les parametres de Flask.
Vous avez aussi un fichier Profile pour un eventuel deploiment sur Heroku.
Il y a aussi requirements.txt avec les bibliothèques python pour faire fonctionner le tout.

Il y a un dossier fonctions avec les fonctions pour les notebooks.
 
Le dossier pingouin_app contient l'application Flask
pingouin_app contient :
    static/css : les fichiers css necessaire pour faire donctionnner les templates
    Static/img des images pour les templates (certaines sont généré par les notebooks)
    static/model_IA : modele entrainé par les notebooks

    templates/ : les templates pour l'application Flask

    form.py :contient les formulaires que flask peut générer.

    utils.py : contient les fonction utiliser par flask pour geneer par exemple une reponse par une IA.

    views.py pour le routage des vues.

    __init__.py : pour generer le modules pour flask.

    Information importante : un fichier .env est fournie avec une clé secrete = "secret". Ne l utiliser pas 
    en production. 

Pour lancer le site en local avec un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate
pip install requirement.txt
python run.py
```
