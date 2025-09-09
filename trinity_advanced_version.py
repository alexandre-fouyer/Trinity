"""
trinity_advanced_version.py (réécrit)
-------------------------------------
Moteur métier "Trinity" utilisé par l'app Streamlit.

• Fournit la classe TrinityBot
• Utilise OpenAI (fallback clé depuis variables d'environnement)
• Interroge l'API PrestaShop en JSON (fallback mini-catalogue si indisponible)
• Construit des URLs produits **propres** via `/{id}-{link_rewrite}.html` si possible,
  sinon fallback vers `index.php?id_product=...&controller=product`.

Dépendances :
  pip install openai requests

Remarque : module pur Python (pas de dépendance Streamlit).
"""
from __future__ import annotations

import os
import re
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


# ================
# 🧠 Utilitaires
# ================

def _clean(s: str) -> str:
    return (s or "").strip()


def _token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _score_match(name: str, query: str) -> int:
    name_l = (name or "").lower()
    score = 0
    for w in set(re.findall(r"\w+", (query or "").lower())):
        if w in name_l:
            score += 1
    return score


def _extract_lang_value(value) -> Optional[str]:
    """PrestaShop peut renvoyer soit une string, soit un mapping/liste par langue."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # cas fréquent : {"language": "slug"} OU {"language": [{"@id":"1","#text":"slug"}, ...]}
        lang = value.get("language")
        if isinstance(lang, str):
            return lang
        if isinstance(lang, list) and lang:
            # on tente #text ou value
            first = lang[0]
            if isinstance(first, dict):
                return first.get("#text") or first.get("value") or first.get("_" )
    return None


def build_product_url(base_url: str, product_id: int, link_rewrite: Optional[str] = None, id_lang: Optional[int] = None) -> str:
    """Construit une URL produit robuste.
    - Si link_rewrite dispo → /{id}-{link_rewrite}.html (PS 1.7/8 routes SEO par défaut)
    - Sinon → fallback /index.php?id_product=..&controller=product (+ id_lang si fourni)
    """
    base = (base_url or "").rstrip("/") or "https://www.levapoteur-discount.fr"
    if link_rewrite:
        # URL propre (SEO)
        return f"{base}/{product_id}-{link_rewrite}.html"
    # Fallback legacy
    if id_lang:
        return f"{base}/index.php?id_product={product_id}&controller=product&id_lang={id_lang}"
    return f"{base}/index.php?id_product={product_id}&controller=product"


# ===================
# 🔌 Client PrestaShop
# ===================

class PrestaClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 15):
        # Fallback secrets/env pour robustesse
        self.base_url = (base_url or os.getenv("PRESTASHOP_URL") or "https://www.levapoteur-discount.fr").rstrip("/")
        self.api_key = api_key or os.getenv("PRESTASHOP_API_KEY") or ""
        self.timeout = timeout

        self.session = requests.Session()
        if self.api_key:
            # Basic Auth: clé en user, mdp vide
            self.session.auth = (self.api_key, "")
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}/api/{endpoint.lstrip('/')}"
        p = {"output_format": "JSON"}
        if params:
            p.update(params)
        resp = self.session.get(url, params=p, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def fetch_products(self, limit: int = 250) -> List[Dict]:
        """Récupère un set minimal de champs pour construire les URLs et l'affichage."""
        # Essai avec display filtré pour limiter la taille
        fields = "[id,id_lang,id_default_image,manufacturer_name,price,name,link_rewrite,active]"
        try:
            data = self._get("products", {"display": fields, "limit": limit, "filter[active]": 1})
        except Exception:
            data = self._get("products", {"limit": limit})
        items = data.get("products", [])

        out: List[Dict] = []
        for it in items:
            try:
                pid = int(it.get("id"))
                name_raw = it.get("name")
                slug_raw = it.get("link_rewrite")
                brand = _clean(it.get("manufacturer_name") or "")
                price = float(it.get("price", 0.0) or 0.0)

                name = _clean(_extract_lang_value(name_raw) or (name_raw if isinstance(name_raw, str) else "Produit"))
                slug = _clean(_extract_lang_value(slug_raw) or "")

                # id_lang est rarement exposé directement, on tente de l'extraire, sinon None
                id_lang = None
                if isinstance(it.get("id_lang"), (int, str)):
                    try:
                        id_lang = int(it.get("id_lang"))
                    except Exception:
                        id_lang = None

                url = build_product_url(self.base_url, pid, link_rewrite=slug or None, id_lang=id_lang)

                out.append({
                    "id": pid,
                    "name": name,
                    "brand": brand,
                    "price": price,
                    "url": url,
                    "slug": slug,
                })
            except Exception:
                continue
        return out

    def fetch_manufacturers(self) -> List[str]:
        try:
            data = self._get("manufacturers", {"display": "[name]", "limit": 250})
            mans = data.get("manufacturers", [])
            return [_clean(_extract_lang_value(m.get("name")) or m.get("name", "")) for m in mans]
        except Exception:
            return []

    def fetch_categories(self) -> List[str]:
        try:
            data = self._get("categories", {"display": "[name,link_rewrite]", "limit": 250})
            cats = data.get("categories", [])
            out = []
            for c in cats:
                nm = _extract_lang_value(c.get("name")) or c.get("name")
                if isinstance(nm, str) and nm.strip():
                    out.append(_clean(nm))
            return out
        except Exception:
            return []


# ==============================
# 🤖 TrinityBot (moteur métier)
# ==============================

class TrinityBot:
    def __init__(
        self,
        prestashop_url: Optional[str] = None,
        prestashop_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialise Trinity avec fallbacks (env).

        - PRESTASHOP_URL, PRESTASHOP_API_KEY
        - OPENAI_API_KEY
        """
        self.prestashop_url = (prestashop_url or os.getenv("PRESTASHOP_URL") or "https://www.levapoteur-discount.fr")
        self.prestashop_key = prestashop_key or os.getenv("PRESTASHOP_API_KEY") or ""

        # Fallback clé OpenAI via env si non fournie par l'UI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError("Clé OpenAI manquante. Renseigne OPENAI_API_KEY (UI, secrets ou env)")

        self.client = OpenAI(api_key=self.openai_api_key)

        self.session_context = SessionContext()
        self.catalog: List[Dict] = []  # [{id,name,brand,price,url,slug}]
        self.manufacturers: List[str] = []
        self.categories: List[str] = []

        self._logger = logging.getLogger("TrinityBot")
        self._logger.setLevel(logging.INFO)

    # --------- Catalogue ---------
    def load_catalog_data(self, force_update: bool = False) -> None:
        """Charge (ou recharge) les données de catalogue depuis PrestaShop.
        Fallback : mini-catalogue local si l'API échoue.
        """
        if self.catalog and not force_update:
            return

        try:
            pc = PrestaClient(self.prestashop_url, self.prestashop_key)
            self.catalog = pc.fetch_products(limit=500)
            self.manufacturers = pc.fetch_manufacturers()
            self.categories = pc.fetch_categories()
            # Déduplication simple par id
            seen = set()
            dedup = []
            for p in self.catalog:
                if p["id"] in seen:
                    continue
                seen.add(p["id"])
                dedup.append(p)
            self.catalog = dedup
        except Exception as e:
            self._logger.warning(f"PrestaShop indisponible, fallback local. Raison: {e}")
            self.catalog = self._fallback_catalog()
            self.manufacturers = sorted(list({p["brand"] for p in self.catalog if p.get("brand")}))
            self.categories = ["Kits débutants", "Pods", "E-liquides fruités", "E-liquides classiques", "Sels de nicotine"]

    def _fallback_catalog(self) -> List[Dict]:
        base = (self.prestashop_url or "https://www.levapoteur-discount.fr").rstrip("/")
        def u(pid: int, slug: str) -> str:
            return build_product_url(base, pid, link_rewrite=slug)
        return [
            {"id": 101, "name": "Kit Débutant Eco 20W", "brand": "LVD", "price": 19.9, "url": u(101, "kit-debutant-eco-20w"), "slug": "kit-debutant-eco-20w"},
            {"id": 102, "name": "Pod Compact 900mAh", "brand": "LVD", "price": 24.9, "url": u(102, "pod-compact-900mah"), "slug": "pod-compact-900mah"},
            {"id": 103, "name": "Pack Démarrage Vape + 3 e-liquides", "brand": "LVD", "price": 29.9, "url": u(103, "pack-demarrage-vape-3-eliquides"), "slug": "pack-demarrage-vape-3-eliquides"},
            {"id": 201, "name": "E-liquide Fraise 10ml 12mg", "brand": "FruitMania", "price": 3.9, "url": u(201, "eliquide-fraise-10ml-12mg"), "slug": "eliquide-fraise-10ml-12mg"},
            {"id": 202, "name": "E-liquide Classic Blond 10ml 12mg", "brand": "Tabac&Co", "price": 3.9, "url": u(202, "eliquide-classic-blond-10ml-12mg"), "slug": "eliquide-classic-blond-10ml-12mg"},
            {"id": 203, "name": "Sel de Nicotine Menthe 20mg", "brand": "FreshNic", "price": 4.9, "url": u(203, "sel-nicotine-menthe-20mg"), "slug": "sel-nicotine-menthe-20mg"},
            {"id": 301, "name": "Clearomiseur MTL 2ml", "brand": "VapoTech", "price": 14.9, "url": u(301, "clearomiseur-mtl-2ml"), "slug": "clearomiseur-mtl-2ml"},
        ]

    # --------- Construction de contexte ---------
    def build_context(self, user_query: str) -> str:
        prof = self.session_context.profile
        stage = self.session_context.conversation_stage
        budget = self.session_context.budget or "non défini"
        already = set(self.session_context.products_shown)

        # Recherche rapide de produits pertinents
        top = self._search_products(user_query, k=10, exclude_ids=already)

        lines = [
            f"PROFIL: {prof}",
            f"STAGE: {stage}",
            f"BUDGET: {budget}",
            f"NB_PRODUITS_MONTRÉS: {len(already)}",
        ]
        if self.categories:
            lines.append("CATÉGORIES_DISPO: " + ", ".join(self.categories[:10]))
        if self.manufacturers:
            lines.append("MARQUES_DISPO: " + ", ".join(self.manufacturers[:10]))
        lines.append("\nPRODUITS_CANDIDATS:")

        for p in top:
            lines.append(f"- [{p['id']}] {p['name']} | {p['brand']} | {p['price']:.2f}€ | {p['url']}")

        return "\n".join(lines)

    # --------- Objections & Qualification ---------
    def handle_objection(self, utterance: str) -> str:
        t = (utterance or "").lower()
        if any(k in t for k in ["cher", "prix", "trop", "budget"]):
            return (
                "Je comprends le budget. On peut partir sur un pack économique fiable, "
                "puis monter en gamme plus tard. Je te propose 1 kit d’entrée + 2 e-liquides 🎯"
            )
        if any(k in t for k in ["compliqué", "complexe", "difficile"]):
            return (
                "Promis on simplifie : un pod simple (tirage auto), une résistance, "
                "et 2 e‑liquides adaptés à ta nicotine. Tu auras 3 étapes maxi."
            )
        return "Je t’écoute : qu’est-ce qui te freine le plus ? Prix, simplicité, autonomie ou saveur ?"

    def generate_qualification_questions(self) -> str:
        q = [
            "Tu fumes ~combien de cigarettes par jour ?",
            "Tu préfères plutôt fruité, menthe, ou goût tabac ?",
            "Tu veux quelque chose de très simple (pod) ou plus évolutif ?",
        ]
        if self.session_context.profile == "TRANSITION":
            q.append("On vise des économies rapides : quel budget mensuel cibles-tu ?")
        return "\n".join(f"- {x}" for x in q)

    # --------- Génération IA (un seul appel) ---------
    def generate_response(self, user_query: str, model: str = "gpt-4o-mini") -> str:
        ctx = self.build_context(user_query)
        sys_prompt = (
            "Tu es Trinity, conseiller expert de Le Vapoteur Discount.\n"
            "Suit ces règles: max 4 produits, jamais reproposer ceux déjà montrés, "
            "packs pour débutants, empathie sur objections, URLs exactes."
        )
        user_prompt = (
            f"Question: {user_query}\n\n{ctx}\n\n"
            "Réponds de façon personnalisée selon PROFIL & STAGE. "
            "Si info manquante, pose 2-3 questions ciblées. Réponse en français."
        )
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.7,
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()

    # --------- Recherche produits interne ---------
    def _search_products(self, query: str, k: int = 10, exclude_ids: Optional[set] = None) -> List[Dict]:
        exclude_ids = exclude_ids or set()
        if not self.catalog:
            return []
        scored = []
        for p in self.catalog:
            if p["id"] in exclude_ids:
                continue
            score = _score_match(p.get("name", ""), query)
            # bonus marque si citée
            if p.get("brand") and p["brand"].lower() in (query or "").lower():
                score += 1
            # heuristique nicotine / type
            if re.search(r"\b(10|12|16|18|20)\s*mg\b", query or "", re.I):
                if re.search(r"(sel|nicotine)", p.get("name", ""), re.I):
                    score += 1
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [p for s, p in scored[:k] if s > 0]
        if not out:  # si rien de pertinent, prendre les moins chers
            out = sorted([p for p in self.catalog if p["id"] not in exclude_ids], key=lambda x: x.get("price", 0))[:k]
        return out

    # --------- Utilitaires publics ---------
    def mark_shown(self, product_ids: List[int]) -> None:
        for pid in product_ids:
            if pid not in self.session_context.products_shown:
                self.session_context.products_shown.append(pid)

    def set_profile(self, profile: str) -> None:
        profile = (profile or "").upper()
        if profile in {"DEBUTANT", "TRANSITION", "AVANCE"}:
            self.session_context.profile = profile

    def set_stage(self, stage: str) -> None:
        stage = (stage or "").lower()
        if stage in {"discovery", "qualification", "recommendation", "closing"}:
            self.session_context.conversation_stage = stage

    def set_budget(self, budget: Optional[str]) -> None:
        self.session_context.budget = budget or None


# Usage rapide (debug):
if __name__ == "__main__":
    # Démo console minimale
    bot = TrinityBot(
        prestashop_url=os.getenv("PRESTASHOP_URL"),
        prestashop_key=os.getenv("PRESTASHOP_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    bot.load_catalog_data()
    print("Produits chargés:", len(bot.catalog))
    ans = bot.generate_response("Je veux un kit pas cher avec e‑liquide fraise 12mg")
    print("\nRéponse:\n", ans)
