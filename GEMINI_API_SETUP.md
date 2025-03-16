# Configuration de l'API Google Gemini

Ce document explique comment configurer votre propre clé API Google Gemini pour l'application de Diagnostic d'Appendicite Pédiatrique.

## Obtenir une clé API Gemini

1. Visitez [Google AI Studio](https://ai.google.dev/)
2. Créez un compte ou connectez-vous avec votre compte Google
3. Naviguez vers la section "API Keys" dans votre compte
4. Cliquez sur "Create API Key" pour générer une nouvelle clé
5. Copiez la clé générée

## Méthodes pour configurer la clé API

Vous pouvez configurer votre clé API Gemini de plusieurs façons :

### 1. Via l'interface web (recommandé)

1. Lancez l'application et connectez-vous
2. Accédez à la page **Paramètres** via le menu de navigation
3. Collez votre clé API dans le champ correspondant
4. Choisissez la méthode de stockage (fichier .env ou session courante)
5. Cliquez sur "Enregistrer"
6. Vous pouvez tester votre clé en cliquant sur "Tester la connexion"

### 2. En utilisant un fichier .env

1. Créez un fichier nommé `.env` à la racine du projet
2. Ajoutez la ligne suivante dans ce fichier :
   ```
   GEMINI_API_KEY=votre_clé_api_ici
   ```
3. Redémarrez l'application si elle est déjà en cours d'exécution

### 3. Via une variable d'environnement système

Définissez la variable d'environnement `GEMINI_API_KEY` dans votre système :

- **Windows** (PowerShell) :
  ```
  $env:GEMINI_API_KEY = "votre_clé_api_ici"
  ```

- **Windows** (Command Prompt) :
  ```
  set GEMINI_API_KEY=votre_clé_api_ici
  ```

- **Linux/macOS** :
  ```
  export GEMINI_API_KEY=votre_clé_api_ici
  ```

## Vérification de l'intégration

Après avoir configuré votre clé API, accédez à la page **Paramètres** pour vérifier que l'intégration avec Google Gemini est active. L'indicateur "Google Gemini disponible" doit être vert.

## Résolution des problèmes courants

### L'API n'est pas détectée comme disponible

- Vérifiez que votre clé API est valide en la testant via la page Paramètres
- Assurez-vous que la bibliothèque `google-generativeai` est installée via pip
- Vérifiez que vous avez accès à Internet et que les services Google sont accessibles depuis votre réseau

### Erreur d'authentification

Si vous rencontrez une erreur d'authentification lors de l'utilisation de l'API :
- Vérifiez que votre clé API est correctement formatée (pas d'espaces supplémentaires)
- Assurez-vous que votre clé API n'a pas expiré
- Générez une nouvelle clé API et mettez à jour votre configuration

### Quota dépassé

Les clés API Google Gemini ont généralement des limites d'utilisation. Si vous dépassez ces limites :
- Attendez que votre quota soit réinitialisé
- Envisagez de mettre à niveau votre niveau d'accès API
- Créez un nouveau projet dans Google AI Studio pour obtenir un nouveau quota
