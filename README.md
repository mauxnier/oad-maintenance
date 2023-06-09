# oad-maintenance
Outils d'aide à la décision pour le pilotage de la maintenance et la réalisation des actions de réparation

https://roland-donat.github.io/cours-rb/ensibs/projet/maintenance/projet.html

# Configuration et Lancement
Pour installer et lancer le projet : Récupérer l'ensemble du projet, installer les dépendances requises si necessaire et lancer le fichier run.py situé dans le dossier "src".

Pour définir un modèle, modifier le fichier run.py pour y ajouter son modèle. Certain modèles sont déja prédéfini (en haut du fichier), il suffit alors de décommenter le modèle voulu.


Le lancement du projet peut s'avérer long nottament à cause de la prédiction. Pour ignorer la prédiction du modèle il suffit de commenter la ligne 68 "modele.evaluate_model()" (désactivé par défaut).

Une fois le projet lancé, une interface web est disponible (à l'adresse http://127.0.0.1:8050/ par défaut).
    
