# Betclic Élite — prédiction (Elo + Poisson)

Prévision des matchs et du classement de la Betclic Élite, saison 2026-2027.
Un moteur Elo adapté au basket estime le niveau de chaque club, un modèle de
points de type Poisson estime comment il marque, et une simulation de
Monte-Carlo rejoue la saison des milliers de fois — saison régulière, play-in
et playoffs compris.

La composition du championnat est désormais connue : l'engagement de l'AS
Monaco a été refusé, Saint-Quentin a été repêché, et la saison a démarré le
25 septembre 2026.

## Lancer

```bash
pip install -r requirements.txt
python -m bepred web
```

Ou, sous Windows, double-cliquer sur `lancer.bat` : le navigateur s'ouvre sur
<http://127.0.0.1:8010>.

Le bouton **↻ Actualiser**, en haut à droite de chaque page, retélécharge les
résultats et réajuste tout le modèle. Il faut compter une dizaine de secondes.

## Les pages

| Page | Ce qu'on y trouve |
|---|---|
| **Projection** | Classement projeté : victoires attendues, top 6, play-in, playoffs, titre, relégation |
| **Composition** | Les seize clubs, et ce que le modèle disait avant que le 16ᵉ soit connu |
| **Rangs** | Probabilité de finir à chaque place, pour chaque club |
| **Calendrier** | Toutes les affiches restantes avec probabilité, écart et total attendus |
| **Match** | Une confrontation au choix, avec la loi jointe des scores |
| **Forces** | Elo, attaque, défense, et volume d'historique disponible par club |
| **Fiabilité** | Validation glissante : le modèle est-il calibré ? |

## En ligne de commande

```bash
python -m bepred projection --sims 20000     # classement projeté, deux scénarios
python -m bepred match Monaco Paris          # une affiche
python -m bepred classement --saison 2025    # classement réel d'une saison
python -m bepred fiabilite                   # validation glissante
python -m bepred regler                      # réglage des hyperparamètres
python -m bepred actualiser                  # retélécharger les résultats
```

## La composition du championnat

La DNCCG, puis la chambre d'appel de la fédération, ont refusé l'engagement de
l'AS Monaco en championnat de France pour 2026-2027. Saint-Quentin, relégué
sportivement à l'issue de 2025-2026, a été repêché à sa place.

Les seize clubs : ASVEL, Boulazac, Chalon, Cholet, Dijon,
Gravelines-Dunkerque, JL Bourg, Le Mans, Limoges, Nancy, Nanterre, Paris,
Pau-Lacq-Orthez, Roanne, Saint-Quentin, Strasbourg.

Jusqu'en août 2026, les deux compositions possibles étaient projetées côte à
côte. L'hypothèse Monaco n'est plus simulée — et pas seulement parce qu'elle
est caduque : Saint-Quentin a joué, et rejouer la saison sans lui demanderait
d'effacer des résultats réels. La page **Composition** conserve ce que le
modèle en disait avant la décision, non corrigé après coup.

Les adresses en `/monaco/…`, publiées avant la décision, redirigent vers la
composition officielle. La liste des clubs n'est plus tenue à la main : dès
qu'un match est joué, elle est déduite des résultats.

## Mettre le site en ligne

Le site se génère entièrement à l'avance : deux scénarios, seize clubs, une
liste finie d'affiches. Aucun serveur n'est nécessaire.

```bash
python -m bepred exporter --sortie site --sims 20000
```

Cela produit ~590 pages (environ 38 Mo) dans `site/` : les deux projections,
les grilles de rangs, les calendriers filtrés par club, les forces, la page
fiabilité, et les 544 affiches possibles (à domicile et sur terrain neutre).
Les liens sont relatifs, le site fonctionne donc aussi bien à la racine d'un
domaine que dans un sous-répertoire.

Pour le vérifier en local :

```bash
python -m http.server 8000 --directory site
```

### Publication automatique sur GitHub Pages

Le dépôt contient `.github/workflows/publier.yml`. Une fois le projet poussé
sur GitHub :

1. **Settings → Pages → Source : GitHub Actions**.
2. C'est tout. Le site se régénère à chaque poussée, chaque matin à 7 h, et à
   la demande depuis l'onglet **Actions**.

Le workflow retélécharge les résultats depuis Wikipedia avant de générer :
c'est lui qui remplace le bouton **Actualiser**. Les CSV de `data/raw/` sont
versionnés et servent de filet de sécurité si Wikipedia est indisponible au
moment de l'exécution.

L'export échoue bruyamment plutôt que de publier un site incomplet : une vue
sans fichier correspondant, une réponse HTTP autre que 200, ou moins de 500
pages générées interrompent la publication.

### Et le bouton Actualiser ?

Il n'a de sens qu'avec un serveur : sur le site statique il est remplacé par
la date de dernière génération. L'application Flask reste utilisable en local
(`python -m bepred web`) quand tu veux recalculer immédiatement, sans attendre
la régénération nocturne.

Si tu préfères héberger l'application Flask elle-même, trois points comptent :
un vrai serveur WSGI (`waitress` ou `gunicorn`), **un seul worker** — l'état du
modèle est une variable globale du processus, plusieurs workers signifieraient
plusieurs modèles indépendants — et une protection sur `/actualiser`, qui
déclenche des téléchargements et un réajustement complet.


## Le modèle

### Elo

Elo classique adapté au basket : le gain de points dépend de la marge de
victoire, amortie par l'écart de niveau (une équipe déjà favorite gagne moins
à s'imposer largement). L'avantage du terrain et l'échelle « points Elo par
point d'écart au score » ne sont pas fixés à la main : ils s'estiment l'un
l'autre sur les données, en quelques itérations. Sur l'historique disponible,
cela donne environ **69 points Elo d'avantage du terrain** et **20 points Elo
pour 1 point d'écart** au tableau d'affichage, soit un avantage du terrain de
**3,4 points**.

Un match gagné en prolongation compte comme un succès d'un point : la
prolongation dit précisément que les deux équipes étaient à égalité à la
sirène.

Entre deux saisons, les notes régressent vers la moyenne. Un club revenant
d'Élite 2 régresse davantage, puis subit une décote de promotion ; un club
jamais rencontré entre sous la moyenne.

### Points (Poisson)

```
lambda_dom = exp(mu + gamma + attaque_dom - defense_ext)
lambda_ext = exp(mu         + attaque_ext - defense_dom)
```

Ajustement par maximum de vraisemblance pondéré (décroissance exponentielle du
poids des matchs anciens, régularisation L2), avec gradient analytique.

Un Poisson pur prédirait un écart-type de √85 ≈ 9 points par équipe, là où la
Betclic Élite en affiche 12,4 : le score d'un match de basket est
sur-dispersé, et les deux scores d'une même rencontre sont corrélés (un match
à haut rythme fait monter les deux). Le modèle ajoute donc :

* un **facteur de rythme partagé** `z ~ Gamma(moyenne 1, variance v)`, qui crée
  la corrélation (mesurée à +0,17 dans les données) ;
* une **dispersion résiduelle** propre à chaque équipe.

Quand les deux s'annulent, on retombe exactement sur deux Poisson
indépendants. C'est l'équivalent, pour le basket, de la correction Dixon-Coles
des petits scores au football.

Le nombre de points marqués en Betclic Élite monte vite — 161 par match en
2023-24, 167 en 2024-25, 173 en 2025-26. Une pondération temporelle, même
courte, garde un pied dans le passé et sous-estime le total. Les forces
relatives restent donc estimées sur tout l'historique, mais le **niveau
d'ensemble est recalé sur la dernière fenêtre observée**.

### Mélange

L'Elo dit *de combien* une équipe est meilleure, le modèle de points dit
*comment* elle marque. Les deux estimations de l'écart attendu sont mélangées,
puis retraduites en couple (λ_dom, λ_ext).

Le poids du Poisson est modulé par le volume de matchs récents disponibles :
un promu sans historique exploitable est jugé presque uniquement sur son Elo
d'arrivée, lui-même prudent par construction.

### Simulation

Chaque saison simulée tire les scores de tous les matchs restants (rythme
partagé, puis dispersion propre), résout les égalités par des prolongations de
cinq minutes, reconstitue le classement, puis déroule :

* le **play-in** : le 7ᵉ reçoit le 8ᵉ, le 9ᵉ reçoit le 10ᵉ, puis le perdant du
  premier match reçoit le vainqueur du second pour la dernière place ;
* les **playoffs** : quarts et demies au meilleur des trois, finale au meilleur
  des cinq.

Les séries se résolvent analytiquement — probabilité exacte de gagner un
best-of-3 ou un best-of-5 compte tenu de l'alternance des terrains — plutôt
que par tirage : moins de bruit pour le même coût.

## Les données

Deux sources, selon l'ancienneté.

**La saison en cours** vient de l'API publique que le site de la LNB interroge
lui-même (`api-prod.lnb.fr`). Les résultats arrivent sans délai, et chaque
match porte sa **date réelle** et son numéro de journée — de quoi rejouer
l'Elo dans le bon ordre.

Cette API ne renvoie qu'une fenêtre glissante d'environ cinq semaines autour
du jour courant et ignore les paramètres de date : les résultats sont donc
**accumulés** dans le cache CSV à chaque passage. Une régénération quotidienne
ne peut rien manquer, et le workflow réenregistre le cache dans le dépôt pour
que l'historique de la saison ne dépende pas d'une seule exécution.

**Les saisons passées** viennent des tableaux croisés de Wikipédia EN (modèle
`sports results`), qui publient la grille complète domicile × extérieur :
2023-24, 2024-25, 2025-26. Les pages antérieures ne publient pas de grille
exploitable. Wikipédia sert aussi de repli si l'API officielle est
indisponible.

Pourquoi ne pas s'en tenir à Wikipédia ? Parce qu'elle n'est alimentée qu'au
bon vouloir des contributeurs. Au 3 octobre 2026, la page 2026-27 existait
avec ses 240 cellules — toutes vides, alors que deux journées avaient été
jouées.

Le cache est dans `data/raw/elite_<année>.csv`, une saison par fichier.

## Ce que le modèle ne sait pas

Ces limites sont réelles, autant les nommer.

* **Le recrutement de l'intersaison.** Le modèle juge les clubs sur leurs
  résultats passés. Un effectif largement remanié en juillet n'est pas vu.
* **Les blessures, les changements d'entraîneur, la coupe d'Europe.** Rien de
  tout cela n'entre dans le modèle.
* **Les playoffs passés.** La source ne publie que la saison régulière : le
  titre 2025-2026 de Monaco n'apporte aucun point Elo.
* **L'ordre des matchs, pour les saisons passées.** La grille Wikipédia donne
  les scores, pas les dates : l'ordre chronologique y est reconstruit par un
  calendrier canonique en ronde. Le *jeu* de matchs joués est exact, seul leur
  ordre est approché. La saison en cours n'est pas concernée — l'API de la LNB
  fournit les dates réelles.
* **Les prolongations de la saison en cours.** L'API ne les signale pas sur la
  route utilisée : ces matchs (environ 4 %) ne sont pas ramenés à 40 minutes
  de jeu, contrairement à ceux des saisons passées.
* **Les départages du règlement.** Le classement simulé départage les clubs à
  égalité de victoires par la différence de points générale, quand le règlement
  LNB passe d'abord par les confrontations directes.
* **La fréquence des prolongations** est légèrement sous-estimée (environ 3 %
  contre 4 % observés) : les fins de match serrées ont une dynamique propre
  qu'un modèle de score ignore.
* **Le total de points** reste sous-estimé d'environ 2,5 points malgré le
  recalage : la hausse du scoring est continue, et un modèle calé sur le passé
  court toujours après.

## Fiabilité mesurée

Validation glissante sur 545 matchs — le modèle est reconstruit sur le seul
passé, puis noté sur les rencontres suivantes.

| | Modèle | Référence |
|---|---|---|
| Score de Brier | **0,197** | 0,243 (toujours l'équipe qui reçoit) |
| Vainqueur trouvé | **70,1 %** | 58,3 % |
| Erreur sur l'écart | 10,7 points | — |
| Erreur sur le total | 13,8 points | — |

La courbe de fiabilité (page **Fiabilité**) montre que les probabilités
annoncées correspondent aux fréquences observées, tranche par tranche.

Les hyperparamètres ont été balayés par validation glissante : les écarts entre
réglages voisins sont faibles au regard du nombre de matchs disponibles. En
particulier, la régression d'intersaison ne peut pas être tranchée par les
données (deux intersaisons seulement dans l'historique) ; la valeur retenue,
25 %, est un a priori raisonnable pour un championnat où les effectifs
changent beaucoup, pas un résultat mesuré.

## Structure

```
bepred/
  config.py      hyperparamètres, zones de classement, saisons chargées
  data.py        sources, normalisation des noms, cache et accumulation
  lnb.py         API officielle de la LNB (saison en cours)
  elo.py         moteur Elo basket
  poisson.py     modèle de points sur-dispersé et corrélé
  model.py       mélange Elo + Poisson
  season.py      classement, matchs restants, simulation, play-in et playoffs
  scenarios.py   les deux compositions possibles et leur comparaison
  backtest.py    validation glissante et calibration
  tune.py        réglage des hyperparamètres
  web.py         interface Flask
  export.py      génération du site statique
data/
  raw/           résultats, une saison par fichier
  params.json    hyperparamètres enregistrés (créé par « bepred regler »)
.github/workflows/
  publier.yml    régénération et publication automatiques du site
```
