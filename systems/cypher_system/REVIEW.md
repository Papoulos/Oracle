# Cypher System : brouillon de pack v0.3 (relu avec le CSRD 2025-08-22)

À placer dans `systems/draft/cypher_system/` (ignoré par la CI).
Source : le fichier `Cypher-System-Reference-Document-2025-08-22.docx` fourni. Un .docx n'a pas de pages :
les références sont des **titres de section**. Pour que le futur contrôle de provenance fonctionne,
exporte le document en PDF.

## Valeurs de ce brouillon (toutes vérifiées dans le CSRD)

| Fichier | Champ | Valeur | Règle | Section du CSRD |
|---|---|---|---|---|
| resolution.json | `multiplier` | 3 | Cible = 3 x difficulté ; réussite si d20 >= cible | How to Play ; Determining Task Difficulty |
| resolution.json | `min_difficulty` / `max_difficulty` | 0 / 10 | 10 degrés de difficulté ; 0 = routine | Task Difficulty (tableau) |
| resolution.json | `auto_success_at` | 0 | Difficulté ramenée à 0 : réussite sans jet | How to Play ; When Do You Roll? |
| resolution.json | `max_total_reduction` | 10 | Au plus 10 pas : compétences 1-2, atouts 1-2, effort 1-6 | Modifying the Difficulty |
| resources.json | pools might / speed / intellect | counter | Trois réserves de caractéristiques | Character Stats ; Pool, Edge, and Effort |
| resources.json | `recovery_triggers` (4) | one_action / ten_minute / one_hour / ten_hour | Les quatre repos successifs | Recovering Points in a Pool |
| resources.json | `pooled_recoveries.recovery_roll` | `d(1, 6) + character.tier`, réparti par le joueur entre les trois réserves | Jet de récupération : 1d6 + palier, à répartir librement | Recovering Points in a Pool |
| resources.json | `recovery_sequences.cypher_rests` | cycle des quatre repos, indice dans `recovery_step` | Le 1er repos demande une action, le 2e dix minutes, le 3e une heure, le 4e dix heures, puis le cycle repart | Recovering Points in a Pool |
| triggers.json | règle `rest_request`, séquence `cypher_rests` | - | "je me repose", "je prends un repos", "je fais une pause", "je reprends mon souffle", "je prends un moment pour me reposer" | Recovering Points in a Pool |

Chemins de la fiche : convention `resources.<id>.current` / `.max`.

## Corrections par rapport à la v0.1

- **Piste de dégâts** : ce n'est pas un état dérivé du nombre de réserves à 0. C'est un état à transitions : une réserve qui tombe à 0 fait descendre d'un cran ; remonter une réserve de 0 à 1+ fait remonter d'un cran ; des dégâts spéciaux (poison, chutes...) la déplacent directement (The Damage Track).
- **Récupération** : ce n'est pas « 4 par jour » mais un **cycle** de 4 repos selon le nombre de repos déjà pris (1 action, 10 min, 1 h, 10 h), puis le cycle recommence (Recovering Points in a Pool).
- **Jets spéciaux** : détail complet ci-dessous (Special Rolls).

## Lacunes (état après les PR de récupération)

| # | Mécanique | Règle (CSRD) | Statut / ce qui manque |
|---|---|---|---|
| G1 | **Récupération par jet** | d6 + palier, répartis librement entre les réserves, sans dépasser le maximum ; cycle de 4 repos | **Couvert** (PR #126 à #128) : formule, répartition par le joueur, cycle de repos. Détection de « je me repose » couverte (PR 5b-1) ; branchement dans l'agent : PR 5b-2 |
| G2 | **Plafonds par catégorie** | Compétences <= 2 pas, atouts <= 2, effort <= 6 | Couvert (PR 4a) |
| G3 | **Coût de l'effort** | 3 points le 1er niveau, 2 les suivants, moins l'Edge de la réserve (Edge utilisé une fois par action et par réserve) ; Edge 3 = 1 niveau gratuit ; +1 par niveau si Impaired ; niveaux <= score d'Effort ; réserve selon la tâche | Couvert (PR 4b) |
| G4 | **Jets spéciaux** | 1 : intrusion du MJ (sans XP) ; 17 / 18 : +1 / +2 dégâts (attaques) ; 19 : effet mineur ou +3 dégâts ; 20 : effet majeur ou +4 dégâts, et le coût en réserve de l'action est remboursé | Le résultat n'expose ni le d20 naturel ni les effets |
| G5 | **Piste de dégâts** | Hale / Impaired / Debilitated / Dead, avec effets (Impaired : effort +1 par niveau, effets spéciaux ignorés ; Debilitated : seulement se déplacer) | Pas de machine à états dans le format ni dans le moteur |
| G6 | **Application des dégâts** | Ordre Might, Speed, Intellect ; l'Armor (nombre fixe) se soustrait ; l'excédent passe à la réserve suivante | Le moteur n'a que des PV |
| G7 | **Jet impossible** | Si même un 20 naturel ne suffit pas, aucun jet n'est fait | Couvert (PR 4a) |
| G8 | **Modificateurs de jet** | Un total de +3 sur le jet se convertit en atout (difficulté -1) | À la charge de l'appelant ; non modélisé |

Le pack a un `triggers.json` avec une seule règle (repos par séquence) ; les `recovery` par réserve restent vides (la récupération passe par la règle partagée) ; la consommation de réserves par phrase n'existe pas pour Cypher (l'effort passe par execute_task).

## Simulation du cycle de repos (graine 11, palier 2)

| Repos | Points (d6 + 2) | Réparti sur | Réserves après | Perdus |
|---|---|---|---|---|
| 1. une action | 6 | Might | 7 / 4 / 10 | 0 |
| 2. dix minutes | 7 | Speed | 7 / 10 / 10 | 1 |
| 3. une heure | 6 | Might | 12 / 10 / 10 | 1 |
| 4. dix heures | 6 | Speed | 12 / 10 / 10 | 6 |

L'indice repasse à 0 après le 4e repos.

## Tests chiffrés existants (simulation, 100 000 jets, graine fixe)

| Cas | Cible | Attendu | Simulé |
|---|---|---|---|
| difficulté 4, aucun ajustement | 12 | 45 % | 44,9 % |
| difficulté 4, compétence -1, effort -1 | 6 | 75 % | 74,9 % |
| difficulté 7, aucun ajustement | 21 | 0 % | 0,0 % |
| difficulté 7, compétence -2, effort -2 | 9 | 60 % | 60,0 % |
| difficulté 2, compétence -2 | 0 | réussite sans jet | OK |
