# OAD maintenance 🚗
Outil d'aide à la décision pour le pilotage de la maintenance et la réalisation des actions de réparation sur des véhicules.
Ce projet utilise les notions de [réseaux bayésiens](https://fr.wikipedia.org/wiki/R%C3%A9seau_bay%C3%A9sien).

https://roland-donat.github.io/cours-rb/ensibs/projet/maintenance/projet.html

# Configuration et Lancement
Pour installer et lancer le projet : Récupérer l'ensemble du projet, installer les dépendances requises si nécessaire et lancer le fichier run.py situé dans le dossier "src".

Pour définir un modèle, modifier le fichier run.py pour y ajouter son modèle. Certains modèles sont déja prédéfinis (en haut du fichier), il suffit alors de décommenter le modèle voulu.


Le lancement du projet peut s'avérer long, notamment à cause de la prédiction. Pour ignorer la prédiction du modèle il suffit de commenter la ligne "modele.evaluate_model()" (désactivé par défaut).

Une fois le projet lancé, une interface web est disponible (à l'adresse http://127.0.0.1:8050/ par défaut).
    
