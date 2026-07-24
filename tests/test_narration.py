import re
import pytest
from agent import Narrator

def test_narrator_prompt_no_numbering():
    """
    Test deterministe : verifie que le prompt systeme du Narrator
    ne contient aucun caractere de numerotation specifique (①②③④⑤)
    ni de motif de ligne commencee par un numero (ex: 1. Perception...)
    dans la partie narrative (avant les regles strictes du bloc resume ou le style).
    """
    narrator = Narrator()
    system_template = narrator.prompt.messages[0].prompt.template

    # Separation pour analyser la structure de la narration elle-meme
    # ou test direct sur l'ensemble du prompt systeme (hors exemples autorises s'il y en a)
    assert not any(char in system_template for char in ["①", "②", "③", "④", "⑤"]), (
        "Le prompt systeme contient encore des chiffres entoures ①②③④⑤"
    )

    # Verifier qu'il n'y a pas de listes numerotees du style "1. Perception"
    # dans le prompt du Narrator.
    # On autorise "1." uniquement s'il n'est pas utilise comme en-tete de structure narrative principale.
    # Ex: pas de "1. Perception", "2. Details..."
    pattern = r"^\s*\d+\.\s+[A-Z\s]+"
    matches = re.findall(pattern, system_template, re.MULTILINE)
    assert not matches, f"Le prompt systeme contient des en-tetes numerotes : {matches}"


PLANTED_SECRET = "la cité engloutie de Xor'thal"

def build_test_context():
    """
    Construit un contexte de scene de test avec un secret cache,
    sur le modele exact du bug des gravures et d'Aethelgard.
    """
    return f"""
SCÈNE COURANTE : Salle de test
Élément présent : des gravures anciennes ornent les parois.
[Connaissance MJ, non révélée au joueur] : ces gravures évoquent en réalité
{PLANTED_SECRET}, mais seul un examen attentif le révèle - le joueur n'a
pas encore examiné ni dechiffre les gravures.
"""

@pytest.mark.integration
def test_resume_ne_fuit_pas_information_non_revelee():
    """
    Test d'integration : effectue un appel reel au LLM (si configure)
    pour verifier que le Narrateur ne fait pas fuiter le secret dans la narration
    ni dans le bloc de resume si le joueur ne l'a pas explicitement decouvert.
    """
    import os
    # S'il n'y a pas de cle OpenAI ni de configuration Ollama fonctionnelle, on skip le test d'integration
    # On peut verifier si l'un des providers est configure et accessible.
    import config
    # On skip si LLM_PROVIDER n'est pas defini ou si c'est la valeur par defaut sans backend running
    # Mais pour etre robuste et ne pas faire de faux negatifs bloquants, on essaie de l'executer
    # et on l'attrape s'il y a une erreur de connexion ou de validation du model.
    try:
        narrator = Narrator()
        # On recupere les instructions standard de l'orchestrateur avec la structure obligatoire
        from agent import RPGAgent
        agent = RPGAgent()
        structure_instructions = agent._build_structure_instructions()

        # Contexte avec instruction de l'orchestrateur
        instructions = f"""
        Action joueur : "J'observe simplement la piece du regard."
        Resultat technique : Aucun jet requis
        {build_test_context()}
        """

        response = narrator.generate_response(
            "J'observe simplement la piece du regard.",
            [],
            instructions + "\n" + structure_instructions
        )
    except Exception as e:
        pytest.skip(f"Backend LLM non configure ou non disponible: {e}")
        return

    assert response is not None

    # Separer la narration et le resume
    parts = response.partition("---")
    narration = parts[0]
    resume = parts[2]

    # On verifie l'absence des chiffres entoures dans le texte reel genere
    assert not any(char in response for char in ["①", "②", "③", "④", "⑤"]), (
        "La reponse du Narrator contient des chiffres entoures ①②③④⑤"
    )

    # Verifier l'absence de lignes numerotees dans la partie narrative (avant ---)
    pattern = r"^\s*\d+\.\s+"
    assert not re.search(pattern, narration, re.MULTILINE), (
        "La narration contient des lignes numerotees (ex: '1. ')"
    )

    # Verifier la non-fuite d'information
    # Le secret "Xor'thal" ne doit apparaitre ni dans la narration ni dans le resume
    assert "xor'thal" not in narration.lower(), (
        f"Le secret '{PLANTED_SECRET}' a fuite dans la narration !"
    )
    assert "xor'thal" not in resume.lower(), (
        f"Le secret '{PLANTED_SECRET}' a fuite dans le bloc resume !"
    )
