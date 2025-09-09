import os
import math
import time
from typing import List

import streamlit as st
from openai import OpenAI

# Importe la logique métier existante (le bot Trinity)
# Assure-toi que le fichier trinity_advanced_version.py est dans le même dossier que cette app.
from trinity_advanced_version import TrinityBot

# ==========================
# 🔧 Utilitaires de chunking
# ==========================

def tokens_estimate(text: str) -> int:
    """Estimation simple (#tokens ≈ #chars / 4). Suffisant pour déclencher le chunking.
    Évite la dépendance à tiktoken pour un déploiement plus simple.
    """
    return max(1, math.ceil(len(text) / 4))


def chunk_text(text: str, max_tokens: int = 1500, overlap_tokens: int = 150) -> List[str]:
    """Découpe un texte en morceaux qui respectent un budget de tokens approximatif.
    On utilise une estimation 1 token ≈ 4 caractères.
    """
    if not text:
        return []

    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_chars  # chevauchement pour le contexte
        if start < 0:
            start = 0
    return chunks


# ======================================
# 🤖 Appels OpenAI (unitaires et par lots)
# ======================================

def _openai_chat(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Effectue un appel Chat Completions simple (sans streaming) et renvoie le texte."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def trinity_response_chunked(
    bot: TrinityBot,
    client: OpenAI,
    model: str,
    user_query: str,
    max_chunk_tokens: int = 1500,
    overlap_tokens: int = 150,
) -> str:
    """Génère la réponse finale en envoyant le contexte par lots à l'API OpenAI.

    Étapes:
      1) Construction du prompt complet (règles Trinity + contexte + question)
      2) Si le prompt est petit => un seul appel
      3) Sinon => découpe du CONTEXTE en lots (chunking) et appels successifs
      4) Synthèse finale sur la base de toutes les analyses partielles
    """
    # On reconstruit le prompt (même logique que generate_response, mais ici on contrôle l'envoi par lots)
    context = bot.build_context(user_query)

    # Gestion "objection" et questions de qualification (on réutilise l'intelligence du bot)
    objection_keywords = ['cher', 'compliqué', 'difficile', 'pas sûr', 'hésite']
    if any(k in user_query.lower() for k in objection_keywords):
        obj_reply = bot.handle_objection(user_query)
        context += f"\n\nRÉPONSE À L'OBJECTION: {obj_reply}"

    if bot.session_context.conversation_stage == "discovery":
        q = bot.generate_qualification_questions()
        if q:
            context += f"\n\nQUESTIONS DE QUALIFICATION:\n{q}"

    system_prompt = f"""
Tu es Trinity, assistant expert Le Vapoteur Discount.

CONTEXTE SESSION:
- Profil: {bot.session_context.profile.value}
- Stage: {bot.session_context.conversation_stage}
- Budget: {bot.session_context.budget or 'non défini'}
- Produits déjà montrés: {len(bot.session_context.products_shown)}

RÈGLES:
1. Si le profil est DEBUTANT, être très pédagogue
2. Si le profil est TRANSITION, insister sur les économies
3. Si des questions de qualification sont présentes, les poser en priorité
4. Ne jamais reproposer les mêmes produits
5. Toujours proposer des bundles/packs pour les débutants
6. Si objection détectée, y répondre avec empathie
7. Utiliser les URLs exactes fournies

COMPORTEMENT SELON LE STAGE:
- discovery: Poser des questions pour qualifier
- qualification: Affiner les besoins
- recommendation: Proposer 3-4 produits avec comparaison
- closing: Aider à finaliser, ne pas reproposer

NE JAMAIS:
- Répéter les salutations
- Proposer plus de 4 produits
- Ignorer les objections
- Reproposer si le client veut commander
""".strip()

    base_user_prompt = (
        "Question: " + user_query + "\n\n" + context + "\n\n"
        "Réponds de manière personnalisée selon le profil et le stage de conversation.\n"
        "Si c'est une objection, traite-la avec empathie.\n"
        "Si manque d'infos, pose les questions de qualification."
    )

    # Si c'est court, un seul call suffit
    if tokens_estimate(base_user_prompt) <= max_chunk_tokens:
        return _openai_chat(client, model, system_prompt, base_user_prompt)

    # Sinon, on découpe UNIQUEMENT la partie contexte pour réduire la taille des lots
    # (on garde l'instruction et la question dans chaque lot)
    instruction_prefix = (
        "Tu vas recevoir des LOTS partiels du contexte.\n"
        "Pour chaque LOT, NE fournis PAS la réponse finale.\n"
        "Renvoie plutôt un JSON compact avec ces clés: \n"
        "- products: liste d'objets {id,name,price,url,is_priority,is_bundle}\n"
        "- facts: liste de puces utiles (max 8)\n"
        "- questions: questions de qualification manquantes (max 3)\n"
        "- objections: liste si pertinente\n"
        "N'inclus pas d'autre prose.\n"
    )

    # Détecte la frontière "Question: ...\n\n" vs CONTEXTE et découpe seulement le contexte lourd
    # Ici, on découpe simplement 'context' (car base_user_prompt = Question + context + consignes)
    chunks = chunk_text(context, max_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)

    partial_jsons: List[str] = []
    progress = st.progress(0.0, text="Analyse par lots en cours…")

    for i, ch in enumerate(chunks, start=1):
        lot_prompt = (
            f"Question: {user_query}\n\n" + instruction_prefix + "\n=== LOT ===\n" + ch
        )
        with st.status(f"Traitement du lot {i}/{len(chunks)}", expanded=False):
            part = _openai_chat(client, model, system_prompt, lot_prompt)
            partial_jsons.append(part)
        progress.progress(i / len(chunks))
        time.sleep(0.1)

    # Synthèse finale
    synth_instructions = (
        "Voici les analyses partielles (JSON) issues de plusieurs LOTS.\n"
        "Fusionne-les intelligemment et rédige maintenant la RÉPONSE FINALE pour l'utilisateur, en suivant STRICTEMENT ces règles:\n"
        "• Maximum 4 produits, jamais les mêmes que déjà montrés.\n"
        "• Si profil débutant -> pédagogie + inclure au moins 1 pack économique.\n"
        "• Si profil transition -> insister sur les économies.\n"
        "• Si objection détectée -> répondre avec empathie.\n"
        "• Utiliser EXACTEMENT les URLs fournies.\n"
        "• Si info manquante, poser 2-3 questions de qualification.\n"
        "• Réponse en français.\n"
    )

    final_prompt = (
        f"Question: {user_query}\n\n" + synth_instructions + "\n\n"
        "=== ANALYSES PARTIELLES ===\n" + "\n---\n".join(partial_jsons)
    )

    return _openai_chat(client, model, system_prompt, final_prompt)


# ==================
# 🎛️ Interface UI
# ==================

st.set_page_config(page_title="Trinity (Streamlit)", page_icon="🤖", layout="wide")

st.title("🤖 Trinity – Conseiller Vente Vape (Streamlit)")

st.markdown(
    """
Cette application Streamlit embarque **Trinity**, votre assistant IA spécialisé pour Le Vapoteur Discount.

**Objectif** :
- Discuter avec Trinity (onglet *Chat Trinity*)
- Analyser de gros contenus en **envoyant par lots** à l'API OpenAI (onglet *Analyse (Chunking)*)

> ⚠️ Ne **hard-code** pas tes clés. Utilise le champ ci-dessous ou `st.secrets`.
"""
)

# -------- Sidebar: Config --------
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    model = st.selectbox(
        "Modèle OpenAI",
        [
            "gpt-4o-mini",  # rapide/éco
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4-turbo",
            "gpt-4",
        ],
        index=0,
    )

    st.divider()
    st.subheader("PrestaShop – Catalogue")
    default_url = "https://www.levapoteur-discount.fr"
    prestashop_url = st.text_input("URL PrestaShop", value=default_url)
    prestashop_key = st.text_input("Clé API PrestaShop", type="password")

    st.caption("Astuce: tu peux mettre tes secrets dans `.streamlit/secrets.toml`.")

# --------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # liste de dicts {role, content}

if "trinity" not in st.session_state:
    st.session_state.trinity = None  # instance TrinityBot

if "catalog_loaded" not in st.session_state:
    st.session_state.catalog_loaded = False

if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    client = None

# ===============
# 🧠 Tabs UI
# ===============

chat_tab, chunk_tab = st.tabs(["💬 Chat Trinity", "📦 Analyse (Chunking)"])

# --------------------
# 💬 Chat Trinity
# --------------------
with chat_tab:
    st.subheader("Chat Trinity")

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        init_click = st.button("Initialiser Trinity", type="primary")
    with colB:
        force_update = st.checkbox("Forcer mise à jour du catalogue", value=False)
    with colC:
        auto_chunk = st.checkbox("Chunking automatique si prompt volumineux", value=True)

    max_chunk_tokens = st.slider("Budget tokens par lot", 800, 4000, 1500, step=100)
    overlap_tokens = st.slider("Chevauchement (tokens)", 50, 400, 150, step=10)

    if init_click:
        if not openai_api_key:
            st.error("Renseigne ta clé OpenAI dans la barre latérale.")
        elif not prestashop_url or not prestashop_key:
            st.error("Renseigne l'URL et la clé API PrestaShop.")
        else:
            # Crée/Met à jour l'instance Trinity
            st.session_state.trinity = TrinityBot(
                prestashop_url=prestashop_url,
                prestashop_key=prestashop_key,
                openai_api_key=openai_api_key,
            )
            st.success("Trinity initialisé ✅")

    # Charger le catalogue si besoin
    if st.session_state.trinity and (force_update or not st.session_state.catalog_loaded):
        with st.spinner("Chargement du catalogue (catégories, marques, produits)…"):
            st.session_state.trinity.load_catalog_data(force_update=force_update)
            st.session_state.catalog_loaded = True
        st.toast("Catalogue prêt ✅", icon="✅")

    # Affiche l'historique
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Zone d'entrée
    user_msg = st.chat_input("Pose ta question (ex: Je veux un kit pas cher, 12mg, fruité)…")

    if user_msg:
        if not client or not openai_api_key:
            st.error("Ajoute ton OpenAI API Key dans la barre latérale.")
        elif not st.session_state.trinity:
            st.error("Clique sur *Initialiser Trinity* d'abord.")
        else:
            # Affiche la question
            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # Génération de la réponse
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    if auto_chunk:
                        answer = trinity_response_chunked(
                            bot=st.session_state.trinity,
                            client=client,
                            model=model,
                            user_query=user_msg,
                            max_chunk_tokens=max_chunk_tokens,
                            overlap_tokens=overlap_tokens,
                        )
                    else:
                        # Fallback: méthode d'origine (un seul appel)
                        answer = st.session_state.trinity.generate_response(user_msg)

                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    placeholder.error(f"Erreur pendant l'appel OpenAI: {e}")

# -----------------------------
# 📦 Analyse (Chunking de texte)
# -----------------------------
with chunk_tab:
    st.subheader("Analyse d'un gros texte – Envoi par lots à l'API OpenAI")
    st.caption(
        "Colle un long texte OU charge un .txt, puis écris ta consigne. L'app découpe le texte en lots, "
        "appelle l'IA pour chaque lot, puis fait une synthèse finale."
    )

    uploaded = st.file_uploader("Importer un fichier .txt (optionnel)", type=["txt"])
    pasted = st.text_area("…ou colle ton texte ici", height=200)
    instruction = st.text_input("Consigne (ex: Résume et propose 3 recommandations produits adaptées)")

    c1, c2, c3 = st.columns(3)
    with c1:
        lot_tokens = st.number_input("Tokens/lot", min_value=800, max_value=4000, value=1500, step=100)
    with c2:
        lot_overlap = st.number_input("Chevauchement", min_value=50, max_value=400, value=150, step=10)
    with c3:
        synth_model = st.selectbox("Modèle (analyse)", [model, "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4-turbo", "gpt-4"], index=0)

    run = st.button("Lancer l'analyse par lots", type="primary")

    if run:
        if not client or not openai_api_key:
            st.error("Ajoute ton OpenAI API Key dans la barre latérale.")
        else:
            text = ""
            if uploaded is not None:
                try:
                    text = uploaded.read().decode("utf-8", errors="ignore")
                except Exception:
                    st.error("Impossible de lire le fichier. Assure-toi que c'est un .txt en UTF-8.")
            if not text:
                text = pasted

            if not text:
                st.warning("Aucun texte fourni.")
            elif not instruction:
                st.warning("Ajoute une consigne.")
            else:
                chunks = chunk_text(text, max_tokens=int(lot_tokens), overlap_tokens=int(lot_overlap))

                st.write(f"Texte total ≈ {tokens_estimate(text)} tokens | {len(chunks)} lot(s)")
                partials = []

                system = (
                    "Tu es Trinity, assistant IA de Le Vapoteur Discount.\n"
                    "Analyse le lot pour répondre à la consigne finale. Ne donne pas la réponse finale ici.\n"
                    "Retourne un JSON compact avec: facts[], products[], leads[], questions[]."
                )

                for i, ch in enumerate(chunks, start=1):
                    with st.status(f"Lot {i}/{len(chunks)} en cours…", expanded=False):
                        lot_prompt = (
                            f"Consigne Finale: {instruction}\n\n"
                            "Analyse UNIQUEMENT ce lot. Extrait l'essentiel utile pour la consigne.\n"
                            "Ne fais pas de redite.\n\n=== LOT ===\n" + ch
                        )
                        out = _openai_chat(client, synth_model, system, lot_prompt)
                        partials.append(out)

                st.success("Tous les lots sont analysés. Synthèse en cours…")

                synth = (
                    f"Consigne Finale: {instruction}\n\n"
                    "Voici des sorties partielles (JSON). Fusionne-les et rédige une seule réponse claire,\n"
                    "structurée, en français, avec au plus 4 recommandations concrètes si pertinent.\n\n"
                    "=== SORTIES PARTIELLES ===\n" + "\n---\n".join(partials)
                )

                final = _openai_chat(client, synth_model, "Tu es Trinity, assistant IA.", synth)
                st.markdown("### Résultat final")
                st.markdown(final)

                with st.expander("Voir les sorties partielles (debug)"):
                    for idx, p in enumerate(partials, start=1):
                        st.markdown(f"**Lot {idx}**\n\n````\n{p}\n````")

st.markdown("---")
st.caption(
    "Astuce: lance l'app avec `streamlit run streamlit_trinity_app.py`.\n"
    "Le chunking est activé par défaut dans l'onglet *Chat Trinity* pour éviter d'envoyer un énorme bloc."
)
