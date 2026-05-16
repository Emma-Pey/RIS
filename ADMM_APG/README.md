# ADMM-APG
_"Efficient Spectral Efficiency Maximization Design for IRS-aided MIMO Systems". Fuying Li, Yajun Wang, Zhuxian Lian, Member, IEEE, and Wen Chen, Senior Member, IEEE. Octobre 2025._

## Librairies nécessaires
- ``numpy``
- ``matplotlib``
- `time`


## Fichiers
- `admm_apg.py`, `layer2.py` et `layer3.py` sont les fichiers qui permettent l'exécution de l'algorithme. 
  - `admm_apg.py` implémente le code global et fait des appels à `layer2.py` pour la mise à jour des variables G, Y, theta et Z. 
  - `layer2.py` implémente la façon dont les différentes variables sont mises à jour et fait appel à `layer3.py`.
  - `layer3.py` implémente le code de plus "bas niveau" (calcul de gradient, algorithme de water-filling...)
- `main.py` définit les variables et appelle `admm_apg.py` pour exécuter l'algorithme. Il implémente aussi la génération des canaux de Rice. 
- `test.py` implémente différentes fonctions qui tracent différentes figures du papier ainsi que d'autres graphes pertinents.

## Lancer le programme
Il faut exécuter le fichier `main.py` pour exécuter l'algorithme brut avec les paramètres définis dans ce même fichier. 

Dans un terminal :

```
py main.py
```

On exécute de la même façon le fichier `test.py`. Toutes les figures seront tracées les unes à la suite des autres. Il peut être utile de commenter (`#`) celles qu'on ne souhaite pas afficher.

## Notes
### Paramètre $\tau$
Le papier ne précisant pas comment le paramètre $\tau^k$ est défini, il a été décidé de l'implémenter comme `tau_k = norm_grad*tau_apg`, pour que le pas soit proportionnel à la taille du gradient. Par défaut, `tau_apg` vaut `0.01`.

L'algorithme est très sensible à ce paramètre. En changeant les valeurs qui décrivent le canal (puissance, bruit, path loss, distances, beta, ...), il se peut que `0.01` ne soit plus adapté et que l'algorithme ne converge plus (si la fonction est trop "bossue", on dépasse le maximum quand le pas est trop grand). C'est notamment le cas pour des SNR initiaux supérieurs ou égaux à 160 dB.

Augmenter `tau_apg` (par exemple `x 10`, `x 100` ou plus) peut permettre de retrouver la convergence. Si cela ne fonctionne pas, on peut aussi tenter de diminuer `rho` (par exemple `/ 10`, `/ 100`), mais sans assurance théorique de converger vers le meilleur résultat...

Cependant, plus `tau_apg` est petit, plus l'algorithme peut converger vite (en 5 itérations notamment). Il est donc nécessaire de le choisir de façon adéquate. 

### SNR
Puisque la puissance à l'émission et la puissance du bruit ne sont utilisés que dans la formule $C=P/(\sigma_n^2 \times Ms)$, j'utilise le SNR à l'émission ($P_{émission}/P_{bruit}$), plutôt que de fixer à la fois $P$ et $\sigma$. Je fixe donc $\sigma$ à 1 dans mes fonctions de test. 