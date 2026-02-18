from fpdf import FPDF
import os

def create_pdf(filename, title, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    for line in content:
        pdf.multi_cell(190, 10, line)
        pdf.ln(2)
    pdf.output(filename)

# Ensure directories exist
os.makedirs("data/codex", exist_ok=True)
os.makedirs("data/intrigue", exist_ok=True)

# Codex PDF
codex_content = [
    "REGLES DE COMBAT: Pour attaquer, lancez un de 20 et ajoutez votre modificateur de Force.",
    "Si le resultat est superieur a la Classe d'Armure (CA) de l'ennemi, vous touchez.",
    "MAGIE: Lancer un sort de soin coute 5 points de Mana. Une boule de feu coute 15 points de Mana.",
    "INVENTAIRE: Chaque personnage peut porter un maximum de 10 objets.",
    "POINTS DE VIE: Si vos PV tombent a 0, vous etes evanoui."
]
create_pdf("data/codex/regles.pdf", "Codex des Regles", codex_content)

# Intrigue PDF
intrigue_content = [
    "LE SECRET DU ROI: Le Roi Elendil n'est pas humain. C'est un dragon polymorphe qui cherche a proteger son tresor cache sous le chateau.",
    "LA QUETE ACTUELLE: La Princesse a disparu. Les villageois pensent qu'elle a ete enlevee par des Goblins, mais elle s'est en realite enfuie pour rejoindre la guilde des voleurs.",
    "LIEU IMPORTANT: La Taverne de l'Ours Boiteux est le repaire secret des rebelles. Le mot de passe est 'Lune Rouge'."
]
create_pdf("data/intrigue/scenario.pdf", "Intrigue: Le Secret du Roi", intrigue_content)

print("Sample PDFs generated successfully.")
