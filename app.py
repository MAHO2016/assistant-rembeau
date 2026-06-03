import os
import random
import requests
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="AssistantRembeau — Entretien Comptable",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .rembeau-header {
        background: linear-gradient(135deg, #E65C00, #2B5EA7);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(230,92,0,0.3);
    }
    .rembeau-logo {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: bold;
        color: #E65C00;
        font-family: 'Rockwell', serif;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .rembeau-title {
        color: #ffffff;
        font-size: 26px;
        font-weight: bold;
        font-family: 'Rockwell', serif;
        margin: 0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .rembeau-subtitle {
        color: #FFD580;
        font-size: 13px;
        font-family: Arial, sans-serif;
        margin: 0;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .badge {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-family: Arial, sans-serif;
        font-weight: 600;
    }
    .badge-orange {
        background: rgba(230,92,0,0.12);
        color: #E65C00;
        border: 1.5px solid #E65C00;
    }
    .badge-blue {
        background: rgba(43,94,167,0.12);
        color: #2B5EA7;
        border: 1.5px solid #2B5EA7;
    }
    .badge-gold {
        background: rgba(255,193,7,0.12);
        color: #b8860b;
        border: 1.5px solid #FFC107;
    }
    .mode-card {
        background: linear-gradient(135deg, #fff8f0, #f0f5ff);
        border: 2px solid #E65C00;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .qcu-card {
        background: #f8faff;
        border: 1.5px solid #2B5EA7;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .rembeau-footer {
        background: linear-gradient(135deg, #E65C00, #2B5EA7);
        padding: 10px 2rem;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.5rem;
    }
    .footer-left {
        color: rgba(255,255,255,0.7);
        font-size: 11px;
        font-family: Arial, sans-serif;
    }
    .footer-right {
        color: #FFD580;
        font-size: 11px;
        font-family: Arial, sans-serif;
        font-weight: bold;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
<div class="rembeau-header">
    <div class="rembeau-logo">AR</div>
    <div>
        <p class="rembeau-title">AssistantRembeau</p>
        <p class="rembeau-subtitle">Votre coach IA pour reussir votre entretien comptable</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="badge badge-orange">🎯 Fiscalite Benin</span>
    <span class="badge badge-blue">📊 Comptabilite OHADA</span>
    <span class="badge badge-gold">🧠 Tests Psychotechniques</span>
    <span class="badge badge-orange">💼 Audit & Finances</span>
</div>
""", unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# === BASE DE QUESTIONS QCU ===
QUESTIONS_QCU = {
    "fiscalite": [
        {
            "question": "Quel est le document de reference en matiere fiscale au Benin ?",
            "options": [
                "A) Le Code du Travail",
                "B) Le Code General des Impots (CGI)",
                "C) Le SYSCOHADA",
                "D) Le Code de Commerce"
            ],
            "reponse": "B",
            "explication": "Le Code General des Impots (CGI) est le document de reference en matiere fiscale. Il faut egalement s'approprier des notes circulaires et de la loi des finances."
        },
        {
            "question": "La TPS (Taxe Professionnelle Synthetique) est due par les contribuables dont le chiffre d'affaires est :",
            "options": [
                "A) Superieur a 100 000 000 FCFA",
                "B) Inferieur ou egal a 50 000 000 FCFA",
                "C) Superieur a 50 000 000 FCFA",
                "D) Inferieur a 10 000 000 FCFA"
            ],
            "reponse": "B",
            "explication": "La TPS est due par les contribuables relevant de l'impot sur les benefices d'affaires mais dont le chiffre d'affaires annuel est inferieur ou egal a cinquante millions (50.000.000) FCFA."
        },
        {
            "question": "Quels impots la TPS regroupe-t-elle ?",
            "options": [
                "A) IBA, Patente, Licences, VPS",
                "B) IS, TVA, IRF, TFU",
                "C) ITS, IRCM, AIB, TVM",
                "D) IBA, TVA, Patente, ITS"
            ],
            "reponse": "A",
            "explication": "La TPS regroupe quatre impots et taxes : l'IBA (Impot sur les Benefices d'Affaires), la contribution des licences, la contribution des patentes et le VPS (Versement Patronal sur les Salaires)."
        },
        {
            "question": "Quel est le taux de la TPS ?",
            "options": [
                "A) 3% des recettes annuelles",
                "B) 10% du benefice net",
                "C) 5% des recettes annuelles",
                "D) 18% du chiffre d'affaires"
            ],
            "reponse": "C",
            "explication": "La TPS est determinee par application d'un taux de 5% aux recettes annuelles. Toutefois, la TPS ne peut etre inferieure a dix mille (10.000) FCFA."
        },
        {
            "question": "A quelle date doit etre souscrite la declaration TPS ?",
            "options": [
                "A) Au plus tard le 31 mars",
                "B) Au plus tard le 30 avril",
                "C) Au plus tard le 30 juin",
                "D) Au plus tard le 31 decembre"
            ],
            "reponse": "B",
            "explication": "Les entreprises relevant du regime du forfait doivent souscrire au plus tard le 30 avril de chaque annee au service des impots territorialement competents, une declaration relative aux activites realisees au cours de l'annee precedente."
        },
        {
            "question": "Quel est le taux de l'AIB pour un prestataire immatricule a l'IFU ?",
            "options": [
                "A) 1%",
                "B) 3%",
                "C) 5%",
                "D) 20%"
            ],
            "reponse": "B",
            "explication": "Le taux de l'AIB est de 3% pour les prestataires immatricules a l'IFU, 5% pour les non immatricules et 20% pour les prestataires non-residents (etrangers)."
        },
        {
            "question": "Les associations sans but lucratif sont-elles soumises a l'IS ?",
            "options": [
                "A) Oui, au taux de 30%",
                "B) Oui, au taux de 15%",
                "C) Non, elles sont exonerees si la gestion est desinteressee",
                "D) Non, jamais sous aucune condition"
            ],
            "reponse": "C",
            "explication": "Les associations et organismes sans but lucratif legalement constitues et dont la gestion est desinteressee sont exoneres de l'IS (Article 4-9 du CGI). La gestion desinteressee implique une direction benevole et le depot d'un rapport d'activite au 30 avril."
        },
        {
            "question": "Quelle est l'echeance de paiement de l'IRF ?",
            "options": [
                "A) Au plus tard le 30 avril",
                "B) Au plus tard le 10 fevrier",
                "C) Au plus tard le 30 juin",
                "D) Au plus tard le 31 decembre"
            ],
            "reponse": "B",
            "explication": "L'IRF (Impot sur le Revenu Foncier) est paye au plus tard le 10 fevrier de chaque annee. Son taux est de 12% du montant brut des loyers."
        },
    ],
    "comptabilite": [
        {
            "question": "Qu'est-ce que la comptabilite ?",
            "options": [
                "A) Un systeme de gestion des ressources humaines",
                "B) Un systeme d'organisation de l'information financiere permettant de presenter une image fidele",
                "C) Un outil de marketing et communication",
                "D) Un systeme de gestion des stocks uniquement"
            ],
            "reponse": "B",
            "explication": "La comptabilite est un systeme d'organisation de l'information financiere permettant de saisir, classer, enregistrer des donnees de base chiffrees et presenter des etats refletant une image fidele du patrimoine, de la situation financiere et du resultat de l'entite."
        },
        {
            "question": "Quels sont les livres comptables obligatoires selon le SYSCOHADA ?",
            "options": [
                "A) Journal, Grand Livre, Balance, Bilan",
                "B) Journal, Grand Livre, Balance generale, Livre d'inventaire",
                "C) Livre de caisse, Livre de banque, Journal, Bilan",
                "D) Grand Livre, Balance, Compte de resultat, Bilan"
            ],
            "reponse": "B",
            "explication": "Les livres comptables obligatoires sont au nombre de 4 : le Journal, le Grand Livre, la Balance generale des comptes et le Livre d'inventaire."
        },
        {
            "question": "Quelles sont les dates comptables importantes ?",
            "options": [
                "A) 01 janvier, 30 juin, 31 decembre, 30 septembre",
                "B) 01 janvier, 31 decembre, 30 avril, 30 juin, 30 septembre",
                "C) 01 mars, 30 juin, 31 octobre, 31 decembre",
                "D) 01 janvier, 31 mars, 30 juin, 31 decembre"
            ],
            "reponse": "B",
            "explication": "Les dates comptables sont : 01 janvier (ouverture), 31 decembre (cloture), 30 avril N+1 (arrete des comptes), 30 juin N+1 (approbation des comptes), 30 septembre (distribution des dividendes)."
        },
        {
            "question": "Qu'est-ce qu'un emballage recuperable ?",
            "options": [
                "A) Un emballage livre avec le produit et non repris",
                "B) Un emballage susceptible d'etre conserve par les tiers et repris par le fournisseur",
                "C) Un emballage usage remis au recyclage",
                "D) Un emballage fourni gratuitement au client"
            ],
            "reponse": "B",
            "explication": "Les emballages recuperables sont les emballages susceptibles d'etre provisoirement conserves par les tiers et que le fournisseur s'engage a reprendre dans des conditions determinees. Ces emballages constituent normalement des immobilisations."
        },
        {
            "question": "Comment sont regroupes les comptes du SYSCOHADA ?",
            "options": [
                "A) Par ordre alphabetique",
                "B) Par categories homogenes appelees classes",
                "C) Par ordre chronologique",
                "D) Par ordre de liquidite"
            ],
            "reponse": "B",
            "explication": "Les comptes du SYSCOHADA sont regroupes par categories homogenes appelees classes. La comptabilite financiere comprend des classes de comptes de situation (classe 1 a 5) et des classes de comptes de gestion (classe 6 a 8)."
        },
        {
            "question": "Qu'est-ce que la regularite comptable ?",
            "options": [
                "A) La tenue quotidienne des livres comptables",
                "B) La conformite aux regles et procedures en vigueur",
                "C) L'exactitude mathematique des calculs",
                "D) La periodicite des etats financiers"
            ],
            "reponse": "B",
            "explication": "La regularite comptable est la conformite aux regles et procedures en vigueur. C'est l'obligation que doit satisfaire toute entite en matiere de tenue, de controle, de presentation et de communication des informations pour assurer l'authenticite des ecritures."
        },
        {
            "question": "Quelle ecriture pour l'acquisition d'un actif ?",
            "options": [
                "A) Credit compte actif / Debit tresorerie",
                "B) Debit compte actif / Credit tresorerie ou passif",
                "C) Debit charges / Credit actif",
                "D) Credit produits / Debit actif"
            ],
            "reponse": "B",
            "explication": "L'acquisition d'un actif est enregistree au bilan en debitant le compte d'actif concerne et en creditant le compte de tresorerie ou le compte de passif utilise pour financer l'acquisition."
        },
    ],
    "psychotechnique": [
        {
            "question": "Completez la suite : 9, 16, 23, 30, 37, (...)",
            "options": [
                "A) 40",
                "B) 43",
                "C) 44",
                "D) 45"
            ],
            "reponse": "C",
            "explication": "Chaque nombre est obtenu en ajoutant 7 au precedent. Donc 37 + 7 = 44. C'est une progression arithmetique de raison +7."
        },
        {
            "question": "Completez la suite : 5, 20, 80, 320, (...)",
            "options": [
                "A) 640",
                "B) 960",
                "C) 1280",
                "D) 1600"
            ],
            "reponse": "C",
            "explication": "Chaque nombre est le produit par 4 du precedent. Donc 320 x 4 = 1280. C'est une progression geometrique de raison x4."
        },
        {
            "question": "Completez la suite : 38, 29, 20, 11, (...)",
            "options": [
                "A) 5",
                "B) 2",
                "C) 3",
                "D) 4"
            ],
            "reponse": "B",
            "explication": "Chaque nombre est obtenu en retranchant 9 du precedent. Donc 11 - 9 = 2. C'est une progression arithmetique decroissante de raison -9."
        },
        {
            "question": "Completez la suite : 486, 162, 54, 18, (...)",
            "options": [
                "A) 9",
                "B) 8",
                "C) 6",
                "D) 3"
            ],
            "reponse": "C",
            "explication": "Chaque nombre est le quotient par 3 du precedent. Donc 18 / 3 = 6. C'est une progression geometrique decroissante de raison /3."
        },
        {
            "question": "Completez la suite : 0, 1, 4, 9, 16, 25, (...)",
            "options": [
                "A) 30",
                "B) 36",
                "C) 49",
                "D) 32"
            ],
            "reponse": "B",
            "explication": "Chaque nombre est le carre de son rang : 0=0^2, 1=1^2, 4=2^2, 9=3^2, 16=4^2, 25=5^2, donc 6^2 = 36."
        },
    ]
}

# === SENTIMENTS ===
SENTIMENTS = {
    "merci": "Avec plaisir ! Je suis la pour vous aider a reussir votre entretien comptable. N'hesitez pas si vous avez d'autres questions.",
    "merci beaucoup": "C'est avec grand plaisir ! Bon courage pour votre entretien. Je reste a votre disposition.",
    "bonjour": "Bonjour ! Je suis AssistantRembeau, votre coach IA pour reussir votre entretien d'embauche au poste de comptable. Comment puis-je vous aider ?",
    "bonsoir": "Bonsoir ! Je suis AssistantRembeau. Comment puis-je vous aider ce soir dans votre preparation ?",
    "bonne nuit": "Bonne nuit ! Continuez vos revisions et n'hesitez pas a revenir demain.",
    "bravo": "Merci pour vos encouragements ! Continuez ainsi, vous etes sur la bonne voie pour reussir votre entretien.",
    "excellent": "Merci ! Je suis la pour vous accompagner jusqu'a la reussite de votre entretien.",
    "parfait": "Merci ! Si vous avez d'autres questions sur la fiscalite, la comptabilite ou l'audit, je suis a votre service.",
    "super": "Merci pour votre retour positif ! Continuez vos revisions avec AssistantRembeau.",
    "ok": "Tres bien ! Y a-t-il autre chose que je puisse faire pour vous ?",
    "au revoir": "Au revoir ! Bon courage pour votre entretien. Vous etes bien prepare !",
    "bonne journee": "Merci, bonne journee a vous ! Bon courage pour vos preparations.",
    "qui es-tu": "Je suis AssistantRembeau, un assistant IA cree par Odilon A. MAFON (Collection MAHO) pour aider les candidats a reussir leur entretien d'embauche au poste de comptable. Je couvre la fiscalite beninoise, la comptabilite OHADA, l'audit et les tests psychotechniques.",
    "que peux-tu faire": "Je peux vous aider a : preparer votre entretien comptable, repondre a vos questions sur la fiscalite beninoise, la comptabilite OHADA, l'audit et les finances, vous entrainer avec des QCU interactifs et vous expliquer les tests psychotechniques.",
    "aide": "Je suis la pour vous aider ! Vous pouvez me poser des questions sur la fiscalite, la comptabilite, l'audit ou demander un QCU pour vous entrainer.",
    "je ne comprends pas": "Pas de probleme ! Reformulez votre question et je ferai de mon mieux pour vous expliquer clairement et simplement.",
    "c'est complique": "Je comprends ! La fiscalite et la comptabilite peuvent sembler complexes. N'hesitez pas a me poser des questions simples, je vous expliquerai etape par etape.",
    "felicitations": "Merci beaucoup ! C'est un plaisir de vous accompagner dans votre preparation.",
    "bonne semaine": "Merci, bonne semaine a vous ! Bonne chance pour vos preparatifs.",
    "bonne annee": "Bonne annee a vous ! Que cette annee soit celle de votre reussite professionnelle.",
}

def detecter_sentiment(question):
    question_lower = question.lower().strip()
    for mot_cle, reponse in SENTIMENTS.items():
        if mot_cle in question_lower:
            return reponse
    return None

FICHIERS_DRIVE = {
    "livre_entretien_comptable.pdf": "VOTRE_ID_DRIVE_ICI",
}

def telecharger_fichiers():
    os.makedirs("documents", exist_ok=True)
    for nom, file_id in FICHIERS_DRIVE.items():
        if file_id == "VOTRE_ID_DRIVE_ICI":
            continue
        chemin = os.path.join("documents", nom)
        if not os.path.exists(chemin):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(url, stream=True)
            with open(chemin, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def charger_modele():
    import shutil
    telecharger_fichiers()
    dossier_docs = "documents"
    dossier_db = "faiss_index_rembeau"
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    if os.path.exists(dossier_db):
        shutil.rmtree(dossier_db)

    documents = []
    if os.path.exists(dossier_docs):
        for fichier in os.listdir(dossier_docs):
            if fichier.endswith('.pdf'):
                chemin = os.path.join(dossier_docs, fichier)
                loader = PyPDFLoader(chemin)
                documents.extend(loader.load())

    if documents:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        morceaux = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(
            documents=morceaux, embedding=embeddings)
        vectorstore.save_local(dossier_db)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    else:
        retriever = None

    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2,
        api_key=OPENAI_API_KEY)
    return llm, retriever


PROMPT = """Tu es AssistantRembeau, un coach IA specialise dans la preparation aux entretiens d'embauche au poste de comptable au Benin et en Afrique francophone.

Tu es base sur le livre "Reussir son entretien d'embauche au poste de comptable" de Odilon A. MAFON (Collection MAHO).

TES DOMAINES D'EXPERTISE :
1. Fiscalite beninoise (CGI, TPS, IBA, TVA, IRF, TFU, TVM, AIB, ITS, IRCM, VPS)
2. Comptabilite OHADA / SYSCOHADA
3. Audit et finances
4. Conseils pour reussir un entretien d'embauche
5. Redaction de CV et lettre de motivation
6. Tests psychotechniques (suites numeriques, logique)

REGLES ABSOLUES :
- Tu reponds UNIQUEMENT sur les sujets comptables, fiscaux, d'audit et d'entretien d'embauche
- Tu donnes des reponses precises et pedagogiques
- Tu encourages toujours le candidat
- Tu cites les articles du CGI quand c'est pertinent
- Tu n'inventes JAMAIS de chiffres ou de taux

REGLE DE CONCISION :
- Reponses courtes et directes pour les questions simples
- Reponses detaillees avec exemples pour les concepts complexes

CONNAISSANCES CLES — FISCALITE BENIN :

TPS (Taxe Professionnelle Synthetique) :
- Due par contribuables avec CA inferieur ou egal a 50 000 000 FCFA
- Regroupe : IBA + Patente + Licences + VPS
- Taux : 5% des recettes annuelles (minimum 10 000 FCFA)
- Declaration : au plus tard le 30 avril
- Paiement : 2 acomptes — 10 fevrier et 10 juin
- Entreprises nouvelles : exonerees les 12 premiers mois

IBA (Impot sur les Benefices d'Affaires) :
- Du par personnes physiques exer ant une activite lucrative
- Determine comme l'IS avec quelques particularites
- Report deficitaire limite a 3 ans
- Amortissement uniquement en mode lineaire

AIB (Acompte d'Impot assis sur le Benefice) :
- 3% pour prestataires immatricules a l'IFU
- 5% pour prestataires non immatricules
- 20% pour prestataires non-residents (etrangers)
- Reversement : au plus tard le 10 du mois suivant

TVA :
- Taux normal : 18%
- Exoneree pour associations a but non lucratif (services benevoles)

IRF (Impot sur Revenu Foncier) :
- Taux : 12% du montant brut des loyers
- Echeance : au plus tard le 10 fevrier

ITS (Impot sur Traitements et Salaires) :
- Taux progressifs : 0% / 10% / 15% / 19% / 30%
- Echeance : 10 du mois suivant

VPS (Versement Patronal sur Salaires) :
- Taux : 4% du salaire brut
- Echeance : 10 du mois suivant

TFU (Taxe Fonciere Unique) :
- 3 a 7% proprietes non baties / 4 a 8% proprietes baties
- Echeances : 50% au 10 fevrier + 50% au 30 avril

TVM (Taxe Vehicules a Moteur) :
- Vehicules 3 roues : 15 000 FCFA
- Transport prive moins de 7 CV : 150 000 FCFA
- Transport prive plus de 7 CV : 200 000 FCFA
- Echeance : 30 avril

IRCM : 15% sur remunerations administrateurs — 10 du mois suivant

IS (Impot sur Societes) :
- Associations a but non lucratif : EXONEREES si gestion desinteressee
- Condition : rapport d'activite au 30 avril

CONNAISSANCES CLES — COMPTABILITE OHADA :

Livres obligatoires (4) : Journal, Grand Livre, Balance generale, Livre d'inventaire
Dates comptables : 01/01 ouverture, 31/12 cloture, 30/04 arrete, 30/06 approbation, 30/09 dividendes
Classes SYSCOHADA : 1-5 comptes de situation, 6-8 comptes de gestion

Principes comptables fondamentaux :
- Continuite d'exploitation
- Permanence des methodes
- Specialisation des exercices
- Prudence
- Cout historique
- Transparence (image fidele)

CONNAISSANCES CLES — ENTRETIEN D'EMBAUCHE :

Questions classiques :
- "Parlez-moi de vous" : presentation structuree (formation + experience + projet)
- "Qualites et defauts" : honnete mais positif, defaut transformable en qualite
- "Pourquoi vous plutot qu'un autre" : competences specifiques + motivation
- "Pretention salariale" : se renseigner sur le marche, donner une fourchette

Conseils :
- Se renseigner sur l'entreprise avant l'entretien
- Arriver 10-15 minutes en avance
- Tenue professionnelle adaptee
- Langage non-verbal positif (contact visuel, posture droite)
- Poser des questions pertinentes a la fin

Contexte extrait des documents :
{context}

Historique :
{historique}

Question : {question}

Reponse pedagogique et encourageante :"""


def generer_reponse(llm, retriever, historique, question):
    if retriever:
        contexte = retriever.invoke(question)
        contexte_formate = "\n\n".join(doc.page_content for doc in contexte)
    else:
        contexte_formate = "Pas de documents charges — reponse basee sur les connaissances integrees."

    historique_formate = ""
    for msg in historique[:-1]:
        role = "Candidat" if msg["role"] == "user" else "Coach"
        historique_formate += f"{role}: {msg['content']}\n"

    prompt_final = PROMPT.format(
        context=contexte_formate,
        historique=historique_formate,
        question=question
    )
    reponse = llm.invoke(prompt_final)
    return reponse.content


# === INITIALISATION ===
llm, retriever = charger_modele()
MAX_MESSAGES = 10
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "conversation"
if "qcu_actif" not in st.session_state:
    st.session_state.qcu_actif = None
if "qcu_repondu" not in st.session_state:
    st.session_state.qcu_repondu = False
if "score_qcu" not in st.session_state:
    st.session_state.score_qcu = {"correct": 0, "total": 0}
if "question_rapide" not in st.session_state:
    st.session_state.question_rapide = None
    
# === CHOIX DU MODE ===
st.markdown("### Choisissez votre mode d'apprentissage :")
col_m1, col_m2 = st.columns(2)
with col_m1:
    if st.button("💬 Mode Conversation — Posez vos questions", use_container_width=True):
        st.session_state.mode = "conversation"
        st.session_state.qcu_actif = None
        st.rerun()
with col_m2:
    if st.button("🧠 Mode QCU — Testez vos connaissances", use_container_width=True):
        st.session_state.mode = "qcu"
        st.rerun()

st.divider()

# ==========================================
# MODE QCU
# ==========================================
if st.session_state.mode == "qcu":
    st.markdown("### 🧠 Mode QCU — Entrainement aux entretiens")

    # Score
    score = st.session_state.score_qcu
    if score["total"] > 0:
        pct = int(score["correct"] / score["total"] * 100)
        st.info(f"Score : **{score['correct']}/{score['total']}** ({pct}%)")

    # Choix categorie
    st.markdown("**Choisissez une categorie :**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏛️ Fiscalite", use_container_width=True):
            questions = QUESTIONS_QCU["fiscalite"]
            st.session_state.qcu_actif = random.choice(questions)
            st.session_state.qcu_repondu = False
            st.rerun()
    with col2:
        if st.button("📊 Comptabilite", use_container_width=True):
            questions = QUESTIONS_QCU["comptabilite"]
            st.session_state.qcu_actif = random.choice(questions)
            st.session_state.qcu_repondu = False
            st.rerun()
    with col3:
        if st.button("🧮 Psychotechnique", use_container_width=True):
            questions = QUESTIONS_QCU["psychotechnique"]
            st.session_state.qcu_actif = random.choice(questions)
            st.session_state.qcu_repondu = False
            st.rerun()

    # Afficher question active
    if st.session_state.qcu_actif:
        q = st.session_state.qcu_actif
        st.markdown("---")
        st.markdown(f"**Question :** {q['question']}")
        st.markdown("")

        for option in q["options"]:
            lettre = option[0]
            if st.button(option, use_container_width=True, key=f"opt_{lettre}"):
                st.session_state.score_qcu["total"] += 1
                if lettre == q["reponse"]:
                    st.session_state.score_qcu["correct"] += 1
                    st.session_state.qcu_repondu = "correct"
                else:
                    st.session_state.qcu_repondu = "incorrect"
                st.rerun()

        if st.session_state.qcu_repondu == "correct":
            st.success(f"✅ Bonne reponse ! Reponse : **{q['reponse']}**")
            st.info(f"💡 **Explication :** {q['explication']}")
            if st.button("➡️ Question suivante", use_container_width=True):
                cat = None
                for k, v in QUESTIONS_QCU.items():
                    if q in v:
                        cat = k
                        break
                if cat:
                    st.session_state.qcu_actif = random.choice(QUESTIONS_QCU[cat])
                    st.session_state.qcu_repondu = False
                    st.rerun()

        elif st.session_state.qcu_repondu == "incorrect":
            st.error(f"❌ Mauvaise reponse. La bonne reponse etait : **{q['reponse']}**")
            st.info(f"💡 **Explication :** {q['explication']}")
            if st.button("➡️ Question suivante", use_container_width=True):
                cat = None
                for k, v in QUESTIONS_QCU.items():
                    if q in v:
                        cat = k
                        break
                if cat:
                    st.session_state.qcu_actif = random.choice(QUESTIONS_QCU[cat])
                    st.session_state.qcu_repondu = False
                    st.rerun()

    # Reset score
    if st.button("🔄 Reinitialiser le score"):
        st.session_state.score_qcu = {"correct": 0, "total": 0}
        st.session_state.qcu_actif = None
        st.rerun()

# ==========================================
# MODE CONVERSATION
# ==========================================
else:
    # Questions frequentes
    if len(st.session_state.messages) == 0:
        st.markdown("**Questions frequentes — cliquez pour poser directement :**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏛️ Qu'est-ce que la TPS ?"):
                st.session_state.question_rapide = "Qu'est-ce que la Taxe Professionnelle Synthetique (TPS) ?"
            if st.button("📊 Livres comptables obligatoires"):
                st.session_state.question_rapide = "Quels sont les livres comptables obligatoires selon le SYSCOHADA ?"
            if st.button("💼 Comment se presenter en entretien ?"):
                st.session_state.question_rapide = "Comment bien repondre a la question 'Parlez-moi de vous' en entretien ?"
        with col2:
            if st.button("💰 Taux AIB Benin"):
                st.session_state.question_rapide = "Quels sont les taux de l'AIB au Benin ?"
            if st.button("📅 Dates comptables importantes"):
                st.session_state.question_rapide = "Quelles sont les dates comptables importantes selon le SYSCOHADA ?"
            if st.button("🎯 Qualites d'un bon comptable"):
                st.session_state.question_rapide = "Quelles sont les qualites necessaires pour etre un bon comptable ?"
        with col3:
            if st.button("🧾 Exoneration IS associations"):
                st.session_state.question_rapide = "Les associations sont-elles exonerees de l'impot sur les societes au Benin ?"
            if st.button("📝 Rediger un bon CV comptable"):
                st.session_state.question_rapide = "Comment rediger un bon CV pour un poste de comptable ?"
            if st.button("⚖️ Difference IBA et IS"):
                st.session_state.question_rapide = "Quelle est la difference entre l'IBA et l'IS au Benin ?"
        st.divider()

    # Traiter question rapide
    if st.session_state.question_rapide:
        question = st.session_state.question_rapide
        st.session_state.question_rapide = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        reponse_sentiment = detecter_sentiment(question)
        if reponse_sentiment:
            reponse = reponse_sentiment
        else:
            with st.spinner("Recherche en cours..."):
                reponse = generer_reponse(llm, retriever, st.session_state.messages, question)
        with st.chat_message("assistant"):
            st.markdown(reponse)
        st.session_state.messages.append({"role": "assistant", "content": reponse})
        st.rerun()

    # Historique messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input
    questions_restantes = MAX_MESSAGES - len([m for m in st.session_state.messages if m["role"] == "user"])
    if questions_restantes <= 3:
        st.warning(f"Il vous reste {questions_restantes} question(s) dans cette session.")
    if len([m for m in st.session_state.messages if m["role"] == "user"]) >= MAX_MESSAGES:
        st.error("Limite de 10 questions atteinte. Cliquez sur 'Nouvelle conversation'.")
    elif question := st.chat_input("Posez votre question sur la comptabilite, la fiscalite ou l'entretien..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        reponse_sentiment = detecter_sentiment(question)
        if reponse_sentiment:
            reponse = reponse_sentiment
        else:
            with st.spinner("Recherche en cours..."):
                reponse = generer_reponse(llm, retriever, st.session_state.messages, question)

        with st.chat_message("assistant"):
            st.markdown(reponse)
        st.session_state.messages.append({"role": "assistant", "content": reponse})

# === BOUTONS BAS DE PAGE ===
st.markdown("")
col1, col2, col3 = st.columns([3, 2, 3])
with col2:
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# === FOOTER ===
st.markdown("""
<div class="rembeau-footer">
    <span class="footer-left">2026 AssistantRembeau — Collection MAHO</span>
    <span class="footer-right">Odilon A. MAFON</span>
</div>
""", unsafe_allow_html=True)
