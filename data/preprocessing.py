"""
Lightweight Stage 1: Data Preprocessing

This module provides a practical preprocessing pipeline aligned with the
methodology: OCR hook (stub), spell/term normalization, ontology-anchored
terminology standardization, temporal alignment to UTC, spatial validation
via facility maps, and schema standardization into a uniform structure RD'.

It is dependency-light (no OCR/PDF libs) to run out-of-the-box. OCR/PDF
extraction hooks are provided as stubs to be replaced in production.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import re

try:
    # Optional import; if ontology available, we leverage it
    from models.ontology import ONTOLOGY
except Exception:  # pragma: no cover
    ONTOLOGY = None


class DataPreprocessor:
    """Implements Stage 1 preprocessing creating a normalized RD' structure."""

    def __init__(self) -> None:
        # Minimal domain lexicon and spelling dictionary (extend as needed)
        self.spelling_map = {
            "instalation": "installation",
            "inspektion": "inspection",
            "measurment": "measurement",
            "tranfser": "transfer",
        }

        # Terminology normalization (synonyms → canonical terms)
        self.terminology_map = {
            # Aerospace/manufacturing
            "bay": "Bay",
            "micrometer": "µm",
            "blade": "Blade",
            # Healthcare units
            "icu": "ICU",
            "micu": "MICU",
            "sicu": "SICU",
        }

        # Facility coordinate maps (examples; extend with real maps)
        self.facility_maps = {
            "aerospace": {
                # Symbolic bays to coordinates (meters)
                "Bay 1": (10.0, 20.0, 0.0),
                "Bay 3": (20.0, 20.0, 0.0),
                "Bay 7": (40.0, 20.0, 0.0),
                "Bay 12": (65.0, 20.0, 0.0),
            },
            "healthcare": {
                "MICU": (10, 20, 3),
                "SICU": (45, 20, 3),
                "CCU": (80, 20, 3),
                "OR": (25, 50, 2),
                "Recovery": (60, 50, 2),
                "Floor": (40, 80, 4),
            },
        }

    # ----------------------------- Public API ----------------------------- #
    def generate_sample_raw_data(self, n: int = 1) -> List[Dict[str, Any]]:
        """Create small synthetic raw inputs across domains to demo Stage 1."""
        docs: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        docs.append({
            "document_id": "doc_aero_001",
            "domain": "aerospace",
            "format": "text",
            "content": {
                "text": "Installation completed in bay 7. Blade Gamma measurment: 3.02 mm.",
            },
            "metadata": {
                "source_type": "inspection_report",
                "timestamp": now,
                "location": "Bay 7",
            }
        })
        docs.append({
            "document_id": "doc_health_001",
            "domain": "healthcare",
            "format": "text",
            "content": {
                "text": "Patient transfer from micu to OR at 10:35 local.",
            },
            "metadata": {
                "source_type": "ehr_note",
                "timestamp": now,
                "location": "MICU",
            }
        })
        return docs[: max(1, min(n, len(docs)))]

    def preprocess_multimodal_data(self, raw_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw documents RD into normalized RD'.

        Returns a dict: {
            'documents': [ { 'document_id', 'domain', 'format', 'content': {
                'text', 'normalized_text', 'structured', 'spatiotemporal'
            }, 'metadata': {...}} ]
        }
        """
        result_docs: List[Dict[str, Any]] = []
        for doc in raw_docs:
            domain = (doc.get("domain") or "").lower() or "aerospace"
            text = self._extract_text(doc)
            normalized_text = self._normalize_text(text)
            standardized_text = self._standardize_terminology(normalized_text)

            # Temporal alignment (UTC ISO 8601)
            ts_src = doc.get("metadata", {}).get("timestamp")
            timestamp_utc = self._to_utc_iso(ts_src)

            # Spatial validation (symbolic → coordinates)
            location_symbol = doc.get("metadata", {}).get("location")
            coords = self._map_location_to_coords(domain, location_symbol)

            # Ontology-driven hints (optional)
            structured = self._ontology_hint_structuring(standardized_text) if ONTOLOGY else {}

            result_docs.append({
                "document_id": doc.get("document_id"),
                "domain": domain,
                "format": doc.get("format", "text"),
                "content": {
                    "text": text,
                    "normalized_text": standardized_text,
                    "structured": structured,
                    "spatiotemporal": {
                        "timestamp_utc": timestamp_utc,
                        "location_symbol": location_symbol,
                        "coordinates": coords,
                    }
                },
                "metadata": doc.get("metadata", {}),
            })

        return {"documents": result_docs}

    # --------------------------- Helper functions ------------------------- #
    def _extract_text(self, doc: Dict[str, Any]) -> str:
        fmt = (doc.get("format") or "text").lower()
        content = doc.get("content", {})
        if fmt == "text" and isinstance(content.get("text"), str):
            return content.get("text", "").strip()
        # Stubs for image/pdf; replace with real OCR/PDF parsers in prod
        if fmt in ("image", "pdf") and isinstance(content.get("file_path"), str):
            return f"[EXTRACTED_TEXT_PLACEHOLDER from {fmt}: {content.get('file_path')}]"
        return ""

    def _normalize_text(self, text: str) -> str:
        # Basic spelling correction and whitespace normalization
        words = re.split(r"(\W+)", text)
        corrected: List[str] = []
        for token in words:
            lower = token.lower()
            if lower in self.spelling_map:
                corrected.append(self.spelling_map[lower])
            else:
                corrected.append(token)
        norm = "".join(corrected)
        return re.sub(r"\s+", " ", norm).strip()

    def _standardize_terminology(self, text: str) -> str:
        def repl(m):
            key = m.group(0).lower()
            return self.terminology_map.get(key, m.group(0))

        # Replace known terms case-insensitively
        pattern = re.compile(r"\b(" + "|".join(map(re.escape, self.terminology_map.keys())) + r")\b", re.I)
        return pattern.sub(repl, text)

    def _to_utc_iso(self, ts: Optional[str]) -> Optional[str]:
        if not ts:
            return None
        try:
            # Accept both naive and Z-suffixed timestamps
            if ts.endswith("Z"):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return None

    def _map_location_to_coords(self, domain: str, symbol: Optional[str]) -> Optional[tuple]:
        if not symbol:
            return None
        domain_map = self.facility_maps.get(domain.lower()) or {}
        # Normalize basic forms like "bay 7" → "Bay 7"
        if domain == "aerospace" and symbol.lower().startswith("bay"):
            parts = symbol.strip().split()
            symbol = f"Bay {parts[-1]}" if parts else symbol
        return domain_map.get(symbol)

    def _ontology_hint_structuring(self, text: str) -> Dict[str, Any]:
        """Very light hinting: detect known relation keywords, entity terms."""
        hints: Dict[str, Any] = {"entities": [], "relations": []}
        # Entities: from ontology classes names present in text
        for name in (ONTOLOGY.entity_classes or {}).keys():
            if name and isinstance(name, str) and re.search(rf"\b{re.escape(name)}\b", text, re.I):
                hints["entities"].append(name)
        # Relations
        for rel in (ONTOLOGY.relationship_types or {}).keys():
            if rel and isinstance(rel, str) and re.search(rf"\b{re.escape(rel)}\b", text, re.I):
                hints["relations"].append(rel)
        return hints


# Shared instance
PREPROCESSOR = DataPreprocessor()


