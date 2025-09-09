"""
trinity_advanced_version.py (version corrigée complète)
--------------------------------------------------------
Moteur métier "Trinity" avec toutes les corrections :
- Tracking des questions pour éviter les répétitions
- Chargement obligatoire du catalogue
- URLs correctes : domain/category/id-slug
- Priorité à la marque "Le Vapoteur Discount"
- Limites configurables pour produits/marques/catégories
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import requests
from openai import OpenAI


# =============================
# 🧱 Contexte de session client
# =============================

@dataclass
class SessionContext:
    profile: str = "DEBUTANT"  # DEBUTANT | TRANSITION | AVANCE
    conversation_stage: str = "discovery"  # discovery | qualification | recommendation | closing
    budget: Optional[str] = None
    products_shown: List[int] = field(default_factory=list)
    # NOUVEAU: Tracking des questions déjà posées pour éviter les répétitions
    questions_asked: Set[str] = field(default_factory=set)
    # NOUVEAU: Stockage des préférences utilisateur extraites
    user_preferences: Dict[str, str] = field(default_factory=dict)


# ================
# 🧠 Utilitaires
# ================

def _clean(s: str) -> str:
    """Nettoie une chaîne."""
    return (s or "").strip()


def _token_estimate(text: str) -> int:
    """Estime le nombre de tokens (1 token ≈ 4 caractères)."""
    return max(1, math.ceil(len(text) / 4))


def _score_match(name: str, query: str) -> int:
    """Score de correspondance entre un nom de produit et une requête."""
    name_l = (name or "").lower()
    score = 0
    for w in set(re.findall(r"\w+", (query or "").lower())):
        if w in name_l:
            score += 1
    return score


def _extract_lang_value(value) -> Optional[str]:
    """Extrait la valeur depuis les structures multilingues PrestaShop."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        lang = value.get("language")
        if isinstance(lang, str):
            return lang
        if isinstance(lang, list) and lang:
            first = lang[0]
            if isinstance(first, dict):
                return first.get("#text") or first.get("value") or first.get("_")
    return None


def build_product_url(base_url: str, product_id: int, category_slug: str, product_slug: str) -> str:
    """
    Construit une URL produit PrestaShop au format correct :
    domain/category/id-slug
    
    Ex: https://www.levapoteur-discount.fr/pods/123-kit-debutant-eco
    """
    base = (base_url or "").rstrip("/") or "https://www.levapoteur-discount.fr"
    
    if category_slug and product_slug:
        # Format standard PrestaShop
        return f"{base}/{category_slug}/{product_id}-{product_slug}"
    
    # Fallback si pas de catégorie
    return f"{base}/produits/{product_id}-{product_slug or 'produit'}"


# ===================
# 📌 Client PrestaShop
# ===================

class PrestaClient:
    """Client pour l'API PrestaShop."""
    
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15):
        self.base_url = (base_url or os.getenv("PRESTASHOP_URL") or "https://www.levapoteur-discount.fr").rstrip("/")
        self.api_key = api_key or os.getenv("PRESTASHOP_API_KEY") or ""
        self.timeout = timeout
        self.session = requests.Session()
        
        if self.api_key:
            # Authentification Basic : clé API comme username, password vide
            self.session.auth = (self.api_key, "")
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Effectue une requête GET sur l'API."""
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"
        p = {"output_format": "JSON"}
        if params:
            p.update(params)
        
        resp = self.session.get(url, params=p, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def fetch_products(self, limit: int = 250) -> List[Dict]:
        """Récupère les produits avec leurs catégories pour construire les URLs."""
        # Champs nécessaires pour l'affichage et les URLs
        fields = "[id,id_default_category,manufacturer_name,price,name,link_rewrite,active,description_short]"
        
        try:
            data = self._get("products", {
                "display": fields,
                "limit": limit,
                "filter[active]": 1
            })
        except Exception as e:
            raise ConnectionError(f"Impossible de récupérer les produits : {e}")
        
        items = data.get("products", [])
        
        # Récupérer le mapping des catégories
        categories_map = self._fetch_categories_map()
        
        out: List[Dict] = []
        for it in items:
            try:
                pid = int(it.get("id"))
                cat_id = int(it.get("id_default_category", 0))
                name_raw = it.get("name")
                slug_raw = it.get("link_rewrite")
                brand = _clean(it.get("manufacturer_name") or "")
                price = float(it.get("price", 0.0) or 0.0)
                
                # Extraire les valeurs multilingues
                name = _clean(_extract_lang_value(name_raw) or (name_raw if isinstance(name_raw, str) else "Produit"))
                slug = _clean(_extract_lang_value(slug_raw) or "")
                
                # Récupérer le slug de la catégorie
                cat_slug = categories_map.get(cat_id, "produits")
                
                # Construire l'URL correcte
                url = build_product_url(self.base_url, pid, cat_slug, slug)
                
                # Détecter si c'est un produit de la marque Le Vapoteur Discount
                is_lvd = any(x in brand.lower() for x in ["vapoteur", "lvd", "discount"])
                
                out.append({
                    "id": pid,
                    "name": name,
                    "brand": brand,
                    "price": price,
                    "url": url,
                    "slug": slug,
                    "category_id": cat_id,
                    "category_slug": cat_slug,
                    "is_lvd": is_lvd,  # Flag pour prioriser les produits LVD
                })
            except Exception:
                continue
        
        return out

    def _fetch_categories_map(self) -> Dict[int, str]:
        """Récupère le mapping id_category -> slug pour construire les URLs."""
        try:
            data = self._get("categories", {
                "display": "[id,link_rewrite]",
                "limit": 500
            })
            cats = data.get("categories", [])
            
            result = {}
            for c in cats:
                try:
                    cat_id = int(c.get("id"))
                    slug_raw = c.get("link_rewrite")
                    slug = _clean(_extract_lang_value(slug_raw) or "")
                    if slug:
                        result[cat_id] = slug
                except Exception:
                    continue
            
            return result
        except Exception:
            return {}

    def fetch_manufacturers(self, limit: int = 100) -> List[str]:
        """Récupère la liste des marques."""
        try:
            data = self._get("manufacturers", {
                "display": "[name]",
                "limit": limit
            })
            mans = data.get("manufacturers", [])
            return [_clean(_extract_lang_value(m.get("name")) or m.get("name", "")) for m in mans]
        except Exception as e:
            raise ConnectionError(f"Impossible de récupérer les marques : {e}")

    def fetch_categories(self, limit: int = 100) -> List[str]:
        """Récupère la liste des catégories."""
        try:
            data = self._get("categories", {
                "display": "[name]",
                "limit": limit
            })
            cats = data.get("categories", [])
            out = []
            for c in cats:
                nm = _extract_lang_value(c.get("name")) or c.get("name")
                if isinstance(nm, str) and nm.strip():
                    out.append(_clean(nm))
            return out
        except Exception as e:
            raise ConnectionError(f"Impossible de récupérer les catégories : {e}")


# ==============================
# 🤖 TrinityBot (moteur métier)
# ==============================

class TrinityBot:
    """
    Bot Trinity - Assistant expert pour Le Vapoteur Discount.
    
    Fonctionnalités principales :
    - Chargement du catalogue PrestaShop obligatoire
    - Tracking des questions pour éviter les répétitions
    - Priorité aux produits de la marque LVD
    - Personnalisation selon le profil client
    """
    
    def __init__(
        self,
        prestashop_url: Optional[str] = None,
        prestashop_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        products_limit: int = 500,
        brands_limit: int = 100,
        categories_limit: int = 100,
    ):
        """
        Initialise Trinity avec limites configurables.
        
        Args:
            prestashop_url: URL de la boutique PrestaShop
            prestashop_key: Clé API PrestaShop
            openai_api_key: Clé API OpenAI
            products_limit: Nombre max de produits à récupérer (ou -1 pour tout)
            brands_limit: Nombre max de marques à récupérer (ou -1 pour tout)
            categories_limit: Nombre max de catégories à récupérer (ou -1 pour tout)
        """
        self.prestashop_url = prestashop_url or os.getenv("PRESTASHOP_URL") or "https://www.levapoteur-discount.fr"
        self.prestashop_key = prestashop_key or os.getenv("PRESTASHOP_API_KEY") or ""
        
        # Clé OpenAI obligatoire
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError("Clé OpenAI manquante. Renseigne OPENAI_API_KEY")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Contexte de session
        self.session_context = SessionContext()
        
        # Catalogue
        self.catalog: List[Dict] = []
        self.manufacturers: List[str] = []
        self.categories: List[str] = []
        
        # Limites configurables (-1 = tout récupérer)
        self.products_limit = products_limit if products_limit != -1 else 99999
        self.brands_limit = brands_limit if brands_limit != -1 else 99999
        self.categories_limit = categories_limit if categories_limit != -1 else 99999
        
        # Logger
        self._logger = logging.getLogger("TrinityBot")
        self._logger.setLevel(logging.INFO)
        
        # Flag pour savoir si le catalogue est chargé
        self.catalog_loaded = False

    def load_catalog_data(self, force_update: bool = False) -> None:
        """
        Charge les données depuis PrestaShop.
        Lève une exception si l'API ne répond pas (pas de fallback).
        """
        if self.catalog_loaded and not force_update:
            return
        
        try:
            pc = PrestaClient(self.prestashop_url, self.prestashop_key)
            
            # Récupération avec les limites configurées
            self.catalog = pc.fetch_products(limit=self.products_limit)
            self.manufacturers = pc.fetch_manufacturers(limit=self.brands_limit)
            self.categories = pc.fetch_categories(limit=self.categories_limit)
            
            # Dédupliquer par ID
            seen = set()
            dedup = []
            for p in self.catalog:
                if p["id"] not in seen:
                    seen.add(p["id"])
                    dedup.append(p)
            self.catalog = dedup
            
            # Trier pour mettre LVD en premier
            self.catalog.sort(key=lambda x: (not x.get("is_lvd", False), x.get("price", 0)))
            
            self.catalog_loaded = True
            self._logger.info(f"Catalogue chargé: {len(self.catalog)} produits, {len(self.manufacturers)} marques, {len(self.categories)} catégories")
            
        except ConnectionError as e:
            # Pas de fallback - on lève l'erreur
            raise ConnectionError("Service temporairement indisponible. L'API PrestaShop ne répond pas.")
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du catalogue: {e}")

    def is_ready(self) -> bool:
        """Vérifie si le bot est prêt (catalogue chargé)."""
        return self.catalog_loaded

    def build_context(self, user_query: str) -> str:
        """
        Construit le contexte pour l'IA en évitant les répétitions.
        Inclut les préférences déjà connues et les questions déjà posées.
        """
        if not self.catalog_loaded:
            raise RuntimeError("Le catalogue n'est pas chargé")
        
        prof = self.session_context.profile
        stage = self.session_context.conversation_stage
        budget = self.session_context.budget or "non défini"
        already = set(self.session_context.products_shown)
        
        # Analyser la requête pour extraire les préférences
        self._extract_preferences(user_query)
        
        # Recherche de produits avec priorité LVD
        top = self._search_products(user_query, k=10, exclude_ids=already)
        
        lines = [
            f"PROFIL: {prof}",
            f"STAGE: {stage}",
            f"BUDGET: {budget}",
            f"NB_PRODUITS_MONTRÉS: {len(already)}",
        ]
        
        # Ajouter les préférences extraites (pour éviter de reposer les questions)
        if self.session_context.user_preferences:
            lines.append("PRÉFÉRENCES_CLIENT (déjà connues):")
            for key, val in self.session_context.user_preferences.items():
                lines.append(f"  - {key}: {val}")
        
        # Questions déjà posées (pour ne pas les répéter)
        if self.session_context.questions_asked:
            lines.append(f"QUESTIONS_DÉJÀ_POSÉES: {', '.join(self.session_context.questions_asked)}")
        
        # Catégories et marques disponibles
        if self.categories[:10]:
            lines.append("CATÉGORIES_DISPO: " + ", ".join(self.categories[:10]))
        if self.manufacturers[:10]:
            lines.append("MARQUES_DISPO: " + ", ".join(self.manufacturers[:10]))
        
        lines.append("\nPRODUITS_CANDIDATS (priorité LVD):")
        
        # Lister les produits avec flag LVD
        for p in top:
            lvd_flag = " [⭐ MARQUE LVD - À PRIVILÉGIER]" if p.get("is_lvd") else ""
            lines.append(f"- [{p['id']}] {p['name']} | {p['brand']}{lvd_flag} | {p['price']:.2f}€ | {p['url']}")
        
        return "\n".join(lines)

    def _extract_preferences(self, user_query: str) -> None:
        """
        Extrait et stocke les préférences de l'utilisateur depuis sa requête.
        Évite de reposer des questions sur les infos déjà données.
        """
        query_lower = user_query.lower()
        
        # Extraction du nombre de cigarettes
        cig_match = re.search(r'(\d+)\s*cigarette', query_lower)
        if cig_match:
            self.session_context.user_preferences["cigarettes_par_jour"] = cig_match.group(1)
            self.session_context.questions_asked.add("cigarettes")
        
        # Extraction des goûts
        if any(w in query_lower for w in ["fruité", "fruit", "fraise", "mangue", "cerise", "pomme", "pêche"]):
            self.session_context.user_preferences["gout"] = "fruité"
            self.session_context.questions_asked.add("gout")
        elif any(w in query_lower for w in ["tabac", "blond", "brun", "classic"]):
            self.session_context.user_preferences["gout"] = "tabac"
            self.session_context.questions_asked.add("gout")
        elif any(w in query_lower for w in ["menthe", "menthol", "frais", "mint"]):
            self.session_context.user_preferences["gout"] = "menthol"
            self.session_context.questions_asked.add("gout")
        
        # Extraction du type souhaité
        if any(w in query_lower for w in ["simple", "facile", "pod", "débutant"]):
            self.session_context.user_preferences["type"] = "pod simple"
            self.session_context.questions_asked.add("type")
        elif any(w in query_lower for w in ["évolutif", "personnalis", "modulable", "avancé"]):
            self.session_context.user_preferences["type"] = "évolutif"
            self.session_context.questions_asked.add("type")
        
        # Extraction nicotine
        nic_match = re.search(r'(\d+)\s*mg', query_lower)
        if nic_match:
            self.session_context.user_preferences["nicotine"] = f"{nic_match.group(1)}mg"
            self.session_context.questions_asked.add("nicotine")
        
        # Extraction budget
        budget_match = re.search(r'(\d+)\s*(?:€|euro)', query_lower)
        if budget_match:
            self.session_context.budget = f"{budget_match.group(1)}€"
            self.session_context.questions_asked.add("budget")

    def handle_objection(self, utterance: str) -> str:
        """Gère les objections courantes avec empathie."""
        t = (utterance or "").lower()
        
        if any(k in t for k in ["cher", "prix", "trop", "budget"]):
            return (
                "Je comprends parfaitement ton souci de budget ! 💰 "
                "Le Vapoteur Discount propose justement des packs économiques parfaits pour débuter. "
                "Tu peux commencer avec un kit complet à moins de 30€ qui inclut tout le nécessaire. "
                "L'investissement initial est vite rentabilisé : tu économiseras environ 150€/mois vs cigarettes !"
            )
        
        if any(k in t for k in ["compliqué", "complexe", "difficile"]):
            return (
                "Aucune inquiétude, c'est beaucoup plus simple qu'il n'y paraît ! 😊 "
                "Les pods modernes sont ultra-simples : tu charges, tu remplis, tu vapes. "
                "Pas de réglages compliqués. Je vais te guider vers les modèles les plus faciles, "
                "avec notre marque Le Vapoteur Discount qui privilégie la simplicité."
            )
        
        return "Dis-moi ce qui te préoccupe le plus : le prix, la simplicité, ou autre chose ?"

    def generate_qualification_questions(self) -> str:
        """
        Génère des questions de qualification NON RÉPÉTÉES.
        Vérifie les questions déjà posées et les préférences déjà connues.
        """
        all_questions = {
            "cigarettes": "Combien de cigarettes fumes-tu par jour environ ?",
            "gout": "Niveau goût, tu préfères fruité 🍓, menthe 🌿, ou tabac classique 🚬 ?",
            "type": "Tu cherches quelque chose de très simple (pod) ou plus évolutif ?",
            "budget": "Quel budget mensuel vises-tu pour la vape ?",
            "nicotine": "Tu connais ton taux de nicotine idéal ou on détermine ensemble ?",
            "autonomie": "L'autonomie est importante pour toi (usage intensif) ?",
        }
        
        # Filtrer les questions : ne pas reposer celles déjà posées ou avec réponse
        questions_to_ask = []
        for key, question in all_questions.items():
            # Ne pas poser si déjà posée ou si on a déjà la préférence
            if key not in self.session_context.questions_asked and key not in self.session_context.user_preferences:
                questions_to_ask.append(question)
                self.session_context.questions_asked.add(key)
                if len(questions_to_ask) >= 3:  # Maximum 3 questions à la fois
                    break
        
        if not questions_to_ask:
            return ""  # Toutes les infos sont déjà connues
        
        return "Pour te proposer le meilleur produit :\n" + "\n".join(f"• {q}" for q in questions_to_ask)

    def generate_response(self, user_query: str, model: str = "gpt-4o-mini") -> str:
        """
        Génère une réponse personnalisée en évitant les répétitions.
        Privilégie les produits LVD et utilise les préférences stockées.
        """
        if not self.catalog_loaded:
            return "⚠️ Le service est temporairement indisponible. Veuillez réessayer dans quelques instants."
        
        ctx = self.build_context(user_query)
        
        # Vérifier si on a des objections
        objection_keywords = ['cher', 'compliqué', 'difficile', 'pas sûr', 'hésite']
        if any(k in user_query.lower() for k in objection_keywords):
            obj_reply = self.handle_objection(user_query)
            ctx += f"\n\nRÉPONSE À L'OBJECTION: {obj_reply}"
        
        # Générer des questions si nécessaire (seulement celles non posées)
        if self.session_context.conversation_stage in ["discovery", "qualification"]:
            q = self.generate_qualification_questions()
            if q:
                ctx += f"\n\nQUESTIONS DE QUALIFICATION (non répétées):\n{q}"
        
        # Prompt système avec instructions claires
        sys_prompt = """Tu es Trinity, expert conseiller de Le Vapoteur Discount.

RÈGLES CRITIQUES:
1. TOUJOURS privilégier et mettre en avant la marque "Le Vapoteur Discount" ou "LVD" (marqués [⭐ MARQUE LVD])
2. NE JAMAIS reposer les mêmes questions - vérifie QUESTIONS_DÉJÀ_POSÉES et PRÉFÉRENCES_CLIENT
3. Si tu as déjà les infos dans PRÉFÉRENCES_CLIENT, passe directement aux recommandations
4. Maximum 4 produits par réponse, avec au moins 1 produit LVD si disponible
5. Utilise les URLs exactes fournies (format: domain/category/id-slug)
6. Sois chaleureux et utilise quelques emojis

ADAPTATION AU PROFIL:
- DEBUTANT: Ultra pédagogue, packs complets LVD, explications simples
- TRANSITION: Focus économies, comparaison cigarettes, ROI rapide
- AVANCE: Technique, performance, nouveautés

IMPORTANT: 
- Si les préférences du client sont déjà dans PRÉFÉRENCES_CLIENT, NE PAS reposer ces questions
- Propose directement des produits adaptés aux préférences connues
- Mets TOUJOURS en avant les produits marqués [⭐ MARQUE LVD] en premier"""

        user_prompt = (
            f"Question: {user_query}\n\n{ctx}\n\n"
            "INSTRUCTION: Réponds de façon personnalisée. "
            "Si tu as déjà les infos nécessaires dans PRÉFÉRENCES_CLIENT, "
            "propose directement des produits SANS reposer de questions. "
            "Privilégie ABSOLUMENT les produits marqués [⭐ MARQUE LVD]."
        )
        
        # Appel à OpenAI
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=700,
        )
        
        # Mettre à jour le stage de conversation si on passe aux recommandations
        response = resp.choices[0].message.content.strip()
        if any(word in response.lower() for word in ["voici", "je te propose", "recommande"]):
            self.session_context.conversation_stage = "recommendation"
        
        return response

    def _search_products(self, query: str, k: int = 10, exclude_ids: Optional[set] = None) -> List[Dict]:
        """
        Recherche de produits avec PRIORITÉ ABSOLUE aux produits LVD.
        Les produits LVD sont toujours mis en avant.
        """
        exclude_ids = exclude_ids or set()
        if not self.catalog:
            return []
        
        scored = []
        for p in self.catalog:
            if p["id"] in exclude_ids:
                continue
            
            score = _score_match(p.get("name", ""), query)
            
            # BONUS FORT pour la marque LVD
            if p.get("is_lvd"):
                score += 10  # Bonus très élevé pour garantir la priorité
            
            # Bonus si marque citée dans la requête
            if p.get("brand") and p["brand"].lower() in (query or "").lower():
                score += 3
            
            # Heuristique nicotine
            if re.search(r"\b(10|12|16|18|20)\s*mg\b", query or "", re.I):
                if re.search(r"(sel|nicotine|\d+mg)", p.get("name", ""), re.I):
                    score += 2
            
            # Bonus goût basé sur les préférences
            query_lower = query.lower()
            name_lower = p.get("name", "").lower()
            
            # Bonus pour correspondance de goût
            if "fruit" in query_lower and any(f in name_lower for f in ["fruit", "fraise", "mangue", "cerise"]):
                score += 2
            if "tabac" in query_lower and any(t in name_lower for t in ["tabac", "blond", "brun", "classic"]):
                score += 2
            if "menth" in query_lower and any(m in name_lower for m in ["menth", "frais", "mint"]):
                score += 2
            
            # Bonus pour type de produit
            if "pod" in query_lower and "pod" in name_lower:
                score += 2
            if "kit" in query_lower and "kit" in name_lower:
                score += 2
            
            scored.append((score, p))
        
        # Tri : score décroissant, puis LVD en priorité, puis prix croissant
        scored.sort(key=lambda x: (-x[0], not x[1].get("is_lvd", False), x[1].get("price", 0)))
        
        # Prendre les meilleurs
        out = [p for s, p in scored[:k] if s > 0]
        
        # S'assurer qu'on a au moins quelques produits LVD dans le résultat
        lvd_products = [p for p in self.catalog 
                       if p.get("is_lvd") and p["id"] not in exclude_ids]
        
        # Ajouter des produits LVD s'il n'y en a pas assez
        lvd_in_out = sum(1 for p in out if p.get("is_lvd"))
        if lvd_in_out < 2 and lvd_products:  # On veut au moins 2 produits LVD
            for p in lvd_products:
                if p not in out:
                    out.insert(0, p)  # Insérer en début de liste
                    if len(out) > k:
                        out.pop()  # Retirer le dernier
                    lvd_in_out += 1
                    if lvd_in_out >= 2:
                        break
        
        return out[:k]

    def mark_shown(self, product_ids: List[int]) -> None:
        """Marque des produits comme déjà montrés pour ne pas les reproposer."""
        for pid in product_ids:
            if pid not in self.session_context.products_shown:
                self.session_context.products_shown.append(pid)

    def set_profile(self, profile: str) -> None:
        """Change le profil du client."""
        profile = (profile or "").upper()
        if profile in {"DEBUTANT", "TRANSITION", "AVANCE"}:
            self.session_context.profile = profile

    def set_stage(self, stage: str) -> None:
        """Change le stage de conversation."""
        stage = (stage or "").lower()
        if stage in {"discovery", "qualification", "recommendation", "closing"}:
            self.session_context.conversation_stage = stage

    def set_budget(self, budget: Optional[str]) -> None:
        """Définit le budget du client."""
        self.session_context.budget = budget or None

    def reset_session(self) -> None:
        """Réinitialise la session (nouveau client)."""
        self.session_context = SessionContext()


# Point d'entrée pour tests
if __name__ == "__main__":
    # Test basique du bot
    bot = TrinityBot(
        prestashop_url=os.getenv("PRESTASHOP_URL"),
        prestashop_key=os.getenv("PRESTASHOP_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        products_limit=100,  # Limiter pour les tests
        brands_limit=50,
        categories_limit=50,
    )
    
    try:
        print("🔄 Chargement du catalogue...")
        bot.load_catalog_data()
        print(f"✅ Catalogue chargé: {len(bot.catalog)} produits")
        
        # Compter les produits LVD
        lvd_count = sum(1 for p in bot.catalog if p.get("is_lvd"))
        print(f"⭐ Produits Le Vapoteur Discount: {lvd_count}")
        
        # Test de conversation
        query = "Je fume 15 cigarettes par jour, j'aime le fruité et je veux un truc simple"
        print(f"\n💬 Question: {query}")
        response = bot.generate_response(query)
        print(f"\n🤖 Réponse:\n{response}")
        
        # Deuxième question (ne devrait pas reposer les mêmes questions)
        query2 = "OK super, montre-moi les produits"
        print(f"\n💬 Question: {query2}")
        response2 = bot.generate_response(query2)
        print(f"\n🤖 Réponse:\n{response2}")
        
    except ConnectionError as e:
        print(f"❌ Erreur de connexion: {e}")
    except Exception as e:
        print(f"❌ Erreur: {e}")