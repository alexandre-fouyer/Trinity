"""
streamlit_app.py (version complète corrigée)
---------------------------------------------
Application Streamlit pour Trinity avec toutes les corrections :
- Chargement obligatoire du catalogue avant utilisation
- Configuration du nombre d'éléments à récupérer
- Chunking optimisé pour réduire les appels API
- Pas de drag & drop, uniquement le chat
- Gestion d'erreur si PrestaShop ne répond pas
"""

import os
import math
import time
import re
from typing import List, Optional

import streamlit as st
from openai import OpenAI

# Import du moteur Trinity corrigé
from trinity_advanced_version import TrinityBot


# ==========================
# 🔧 Utilitaires de chunking
# ==========================

def tokens_estimate(text: str) -> int:
    """Estimation simple des tokens (1 token ≈ 4 caractères)."""
    return max(1, math.ceil(len(text) / 4))


def chunk_text(text: str, max_tokens: int = 1000, overlap_tokens: int = 100) -> List[str]:
    """
    Découpe intelligente du texte en morceaux plus petits.
    Permet de réduire le volume envoyé à OpenAI.
    """
    if not text:
        return []
    
    # Conversion tokens -> caractères
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
        
        # Chevauchement pour maintenir le contexte
        start = end - overlap_chars
        if start < 0:
            start = 0
    
    return chunks


# ======================================
# 🤖 Appels OpenAI optimisés
# ======================================

def openai_chat_single(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 800
) -> str:
    """Effectue un appel simple à l'API OpenAI."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Erreur OpenAI: {e}")


def trinity_response_optimized(
    bot: TrinityBot,
    client: OpenAI,
    model: str,
    user_query: str,
    max_chunk_tokens: int = 1000,
) -> str:
    """
    Génère la réponse Trinity en optimisant les appels API.
    Utilise le chunking si le contexte est trop grand.
    """
    
    # Vérifier que le bot est prêt
    if not bot.is_ready():
        return "⚠️ Le service n'est pas encore prêt. Veuillez initialiser Trinity."
    
    # Construire le contexte
    try:
        context = bot.build_context(user_query)
    except RuntimeError as e:
        return f"⚠️ {e}"
    
    # Gérer les objections si détectées
    objection_keywords = ['cher', 'compliqué', 'difficile', 'pas sûr', 'hésite', 'doute']
    if any(k in user_query.lower() for k in objection_keywords):
        obj_reply = bot.handle_objection(user_query)
        context += f"\n\nRÉPONSE OBJECTION PRÉPARÉE: {obj_reply}"
    
    # Ajouter les questions de qualification si nécessaire
    if bot.session_context.conversation_stage in ["discovery", "qualification"]:
        q = bot.generate_qualification_questions()
        if q:
            context += f"\n\n{q}"
    
    # Prompt système compact mais complet
    system_prompt = """Tu es Trinity, expert conseiller Le Vapoteur Discount.

PRIORITÉS:
1. Mettre en avant la marque "Le Vapoteur Discount" (LVD) - produits marqués [⭐ MARQUE LVD]
2. Ne JAMAIS reposer des questions déjà dans QUESTIONS_DÉJÀ_POSÉES ou PRÉFÉRENCES_CLIENT
3. Maximum 4 produits, URLs exactes fournies
4. Personnaliser selon le profil (DEBUTANT/TRANSITION/AVANCE)"""
    
    # Si le contexte est petit, un seul appel suffit
    estimated_tokens = tokens_estimate(context)
    
    if estimated_tokens <= max_chunk_tokens:
        # Appel direct sans chunking
        user_prompt = (
            f"Question client: {user_query}\n\n"
            f"Contexte et données:\n{context}\n\n"
            "Instructions: Réponds de manière personnalisée. "
            "Si les infos sont dans PRÉFÉRENCES_CLIENT, propose directement des produits. "
            "Privilégie TOUJOURS les produits [⭐ MARQUE LVD]."
        )
        return openai_chat_single(client, model, system_prompt, user_prompt)
    
    # Sinon, on fait du chunking pour économiser les tokens
    chunks = chunk_text(context, max_tokens=max_chunk_tokens, overlap_tokens=100)
    
    # Analyser chaque chunk séparément
    analyses = []
    progress_bar = st.progress(0.0)
    
    for i, chunk in enumerate(chunks, 1):
        progress_bar.progress(i / len(chunks), f"Analyse {i}/{len(chunks)}...")
        
        chunk_prompt = f"""Question client: {user_query}

Analyse ce contexte partiel et extrais UNIQUEMENT:
1. Produits LVD pertinents (avec [⭐])
2. Autres produits pertinents
3. Préférences client détectées
4. Questions nécessaires NON répétées

CONTEXTE PARTIEL:
{chunk}

Retourne un JSON compact avec: products[], preferences[], questions[]"""
        
        try:
            analysis = openai_chat_single(
                client, model, system_prompt, chunk_prompt,
                temperature=0.5, max_tokens=400
            )
            analyses.append(analysis)
        except Exception as e:
            st.warning(f"Erreur chunk {i}: {e}")
    
    progress_bar.empty()
    
    # Synthèse finale des analyses
    synthesis_prompt = f"""Question originale du client: {user_query}

Analyses des données:
{chr(10).join(analyses)}

GÉNÈRE LA RÉPONSE FINALE:
- Privilégier ABSOLUMENT les produits marqués [⭐ MARQUE LVD]
- Maximum 4 produits total
- Si les préférences sont connues, ne PAS reposer de questions
- URLs exactes des produits
- Ton chaleureux avec quelques emojis
- En français"""
    
    return openai_chat_single(client, model, system_prompt, synthesis_prompt)


# ==================
# 🎛️ Interface UI
# ==================

# Configuration de la page
st.set_page_config(
    page_title="Trinity - Le Vapoteur Discount",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🤖 Trinity - Conseiller Expert Le Vapoteur Discount")
st.markdown("*Votre assistant IA spécialisé pour arrêter de fumer*")

# Initialisation des états de session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trinity" not in st.session_state:
    st.session_state.trinity = None

if "catalog_loaded" not in st.session_state:
    st.session_state.catalog_loaded = False

if "init_error" not in st.session_state:
    st.session_state.init_error = None

# -------- Sidebar Configuration --------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Section API Keys
    with st.expander("🔑 Clés API", expanded=True):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Requis pour le fonctionnement du chatbot",
            placeholder="sk-..."
        )
        
        model = st.selectbox(
            "Modèle OpenAI",
            ["gpt-4o-mini", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="gpt-4o-mini est plus rapide et économique"
        )
    
    # Section PrestaShop
    with st.expander("🛍️ PrestaShop", expanded=True):
        prestashop_url = st.text_input(
            "URL PrestaShop",
            value="https://www.levapoteur-discount.fr",
            help="URL de votre boutique PrestaShop"
        )
        
        prestashop_key = st.text_input(
            "Clé API PrestaShop",
            type="password",
            help="Clé d'accès à l'API PrestaShop",
            placeholder="Votre clé API..."
        )
    
    # Section Limites de récupération
    with st.expander("📊 Limites de récupération", expanded=False):
        st.markdown("*Configurez le nombre d'éléments à récupérer*")
        
        # Produits
        col1, col2 = st.columns([1, 2])
        with col1:
            fetch_all_products = st.checkbox("Tout", key="all_products")
        with col2:
            if fetch_all_products:
                products_limit = -1
                st.info("Tous les produits")
            else:
                products_limit = st.number_input(
                    "Nb produits",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=50,
                    label_visibility="collapsed"
                )
        
        # Marques
        col1, col2 = st.columns([1, 2])
        with col1:
            fetch_all_brands = st.checkbox("Tout", key="all_brands")
        with col2:
            if fetch_all_brands:
                brands_limit = -1
                st.info("Toutes les marques")
            else:
                brands_limit = st.number_input(
                    "Nb marques",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    label_visibility="collapsed"
                )
        
        # Catégories
        col1, col2 = st.columns([1, 2])
        with col1:
            fetch_all_categories = st.checkbox("Tout", key="all_cats")
        with col2:
            if fetch_all_categories:
                categories_limit = -1
                st.info("Toutes les catégories")
            else:
                categories_limit = st.number_input(
                    "Nb catégories",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    label_visibility="collapsed"
                )
    
    # Section Optimisation
    with st.expander("🔧 Optimisation", expanded=False):
        max_chunk_tokens = st.slider(
            "Tokens max par chunk",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Plus bas = économie de tokens, plus haut = meilleur contexte"
        )
        
        st.caption("💡 Réduire les tokens diminue les coûts API")
    
    st.divider()
    
    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        init_button = st.button(
            "🚀 Initialiser",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.catalog_loaded,
            help="Charger le catalogue depuis PrestaShop"
        )
    
    with col2:
        if st.session_state.catalog_loaded:
            reset_button = st.button(
                "🔄 Réinitialiser",
                use_container_width=True,
                help="Recharger le catalogue"
            )
            if reset_button:
                st.session_state.catalog_loaded = False
                st.session_state.trinity = None
                st.session_state.messages = []
                st.rerun()

# Client OpenAI
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# ===============
# 🚀 Initialisation
# ===============

if init_button:
    # Vérifications
    errors = []
    if not openai_api_key:
        errors.append("Clé OpenAI manquante")
    if not prestashop_url:
        errors.append("URL PrestaShop manquante")
    if not prestashop_key:
        errors.append("Clé API PrestaShop manquante")
    
    if errors:
        st.error("❌ " + " | ".join(errors))
    else:
        # Tentative d'initialisation
        progress_container = st.empty()
        
        with progress_container.container():
            with st.spinner("🔄 Connexion à PrestaShop..."):
                try:
                    # Créer l'instance Trinity avec les limites configurées
                    st.session_state.trinity = TrinityBot(
                        prestashop_url=prestashop_url,
                        prestashop_key=prestashop_key,
                        openai_api_key=openai_api_key,
                        products_limit=products_limit,
                        brands_limit=brands_limit,
                        categories_limit=categories_limit,
                    )
                    
                    # Message de progression
                    st.info("📦 Chargement du catalogue en cours...")
                    
                    # Charger le catalogue (obligatoire, pas de fallback)
                    st.session_state.trinity.load_catalog_data()
                    st.session_state.catalog_loaded = True
                    st.session_state.init_error = None
                    
                    # Stats de chargement
                    catalog = st.session_state.trinity.catalog
                    lvd_products = [p for p in catalog if p.get("is_lvd")]
                    
                    st.success(f"""✅ **Trinity initialisé avec succès!**
                    
📊 **Catalogue chargé:**
- 📦 {len(catalog)} produits
- 🏷️ {len(st.session_state.trinity.manufacturers)} marques
- 📁 {len(st.session_state.trinity.categories)} catégories
- ⭐ {len(lvd_products)} produits Le Vapoteur Discount
                    """)
                    
                    # Afficher quelques produits LVD en exemple
                    if lvd_products[:3]:
                        st.info("🌟 **Produits LVD disponibles:** " + 
                               ", ".join([p["name"] for p in lvd_products[:3]]) + "...")
                    
                    time.sleep(2)
                    st.rerun()
                    
                except ConnectionError as e:
                    st.session_state.init_error = "Service temporairement indisponible. L'API PrestaShop ne répond pas."
                    st.error(f"❌ {st.session_state.init_error}")
                    
                except Exception as e:
                    st.session_state.init_error = str(e)
                    st.error(f"❌ Erreur: {e}")

# ===============
# 💬 Zone de chat
# ===============

# Vérifier l'état du système
if st.session_state.init_error:
    # Erreur persistante
    st.error(f"⚠️ {st.session_state.init_error}")
    st.info("💡 Vérifiez votre configuration PrestaShop et réessayez.")
    
elif not st.session_state.catalog_loaded:
    # Pas encore initialisé
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👆 **Cliquez sur 'Initialiser' dans la barre latérale pour commencer**")
    
    # Exemples de questions
    st.markdown("---")
    st.subheader("💡 Exemples de questions pour Trinity")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🚭 Arrêt du tabac:**
        - "Je fume 20 cigarettes par jour, aidez-moi"
        - "Je veux arrêter mais c'est compliqué"
        - "Quel budget pour remplacer la cigarette?"
        """)
    
    with col2:
        st.markdown("""
        **🛍️ Choix de produits:**
        - "Kit simple pour débuter, goût fruité"
        - "Pod pas cher avec bonne autonomie"
        - "E-liquide 12mg saveur tabac blond"
        """)
    
else:
    # Système prêt - Afficher les stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Produits", len(st.session_state.trinity.catalog))
    with col2:
        st.metric("🏷️ Marques", len(st.session_state.trinity.manufacturers))
    with col3:
        st.metric("📁 Catégories", len(st.session_state.trinity.categories))
    with col4:
        lvd_count = len([p for p in st.session_state.trinity.catalog if p.get("is_lvd")])
        st.metric("⭐ Produits LVD", lvd_count)
    
    st.divider()
    
    # Afficher l'historique de conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Zone d'entrée utilisateur
    user_input = st.chat_input(
        "Posez votre question à Trinity...",
        key="chat_input"
    )
    
    if user_input:
        if not client:
            st.error("❌ Clé OpenAI manquante. Ajoutez-la dans la configuration.")
        else:
            # Ajouter le message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Générer et afficher la réponse
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                try:
                    with st.spinner("Trinity réfléchit..."):
                        # Générer la réponse avec chunking optimisé
                        response = trinity_response_optimized(
                            bot=st.session_state.trinity,
                            client=client,
                            model=model,
                            user_query=user_input,
                            max_chunk_tokens=max_chunk_tokens,
                        )
                    
                    # Afficher la réponse
                    response_placeholder.markdown(response)
                    
                    # Sauvegarder dans l'historique
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Extraire et marquer les produits montrés (pour ne pas les reproposer)
                    product_ids = re.findall(r'\[(\d+)\]', response)
                    if product_ids:
                        st.session_state.trinity.mark_shown([int(pid) for pid in product_ids])
                        
                        # Petit feedback visuel
                        with st.sidebar:
                            st.success(f"✅ {len(product_ids)} produits proposés")
                    
                except Exception as e:
                    error_msg = f"❌ Erreur: {e}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("""
    💡 **Trinity** utilise le chunking intelligent pour optimiser les appels API
    
    🌟 Priorité aux produits **Le Vapoteur Discount**
    """)