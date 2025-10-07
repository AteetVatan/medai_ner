"""
Medical Named Entity Recognition Service for German text.

This module provides the core NER functionality, combining multiple models
and performing entity extraction, normalization, and deduplication.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime

from typing import ForwardRef

if not hasattr(ForwardRef, "_orig_evaluate"):
    ForwardRef._orig_evaluate = ForwardRef._evaluate

    def _patched_evaluate(self, *args, **kwargs):
        """
        Wrap the original _evaluate, ignoring extra recursive_guard argument if present.
        Accept either (globalns, localns), or (globalns, localns, recursive_guard), or keywords.
        """
        try:
            return ForwardRef._orig_evaluate(self, *args, **kwargs)
        except TypeError as e:
            # Possibly missing recursive_guard kwarg or wrong signature
            # Try calling with only first two args
            if len(args) >= 2:
                return ForwardRef._orig_evaluate(self, args[0], args[1])
            # fallback: call orig with no args
            return ForwardRef._orig_evaluate(self)

    ForwardRef._evaluate = _patched_evaluate

# Now import spaCy
import spacy
from spacy.tokens import Doc, Span
from spacy.pipeline import EntityRuler
from spacy.matcher import Matcher

from spacy.language import Language

from config import ServiceConfig, ENTITY_MAPPING, ICD_CONFIG
from model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class MedicalEntity:
    """Represents a detected medical entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_text: Optional[str] = None
    icd_code: Optional[str] = None
    icd_description: Optional[str] = None
    source_model: str = "spacy"
    category: Optional[str] = None


class ICDDictionary:
    """ICD-10/ICD-11 dictionary for medical entity mapping."""

    def __init__(self):
        self.icd_codes = {}
        self._load_icd_dictionary()

    def _load_icd_dictionary(self):
        """Load ICD-10/ICD-11 codes and descriptions."""
        # In production, this would load from a comprehensive database
        # For MVP, we'll use a subset of common German medical terms
        self.icd_codes = {
            # Common symptoms
            "kopfschmerzen": {
                "code": "R51",
                "description": "Kopfschmerz",
                "category": "symptom",
            },
            "schwindel": {
                "code": "R42",
                "description": "Schwindel und Gangunsicherheit",
                "category": "symptom",
            },
            "müdigkeit": {
                "code": "R53",
                "description": "Unwohlsein und Ermüdung",
                "category": "symptom",
            },
            "fieber": {"code": "R50", "description": "Fieber", "category": "symptom"},
            "schmerzen": {
                "code": "R52",
                "description": "Schmerz",
                "category": "symptom",
            },
            "übelkeit": {
                "code": "R11",
                "description": "Übelkeit und Erbrechen",
                "category": "symptom",
            },
            # Common diagnoses
            "hypertension": {
                "code": "I10",
                "description": "Essentielle Hypertonie",
                "category": "diagnosis",
            },
            "diabetes": {
                "code": "E11",
                "description": "Diabetes mellitus, Typ 2",
                "category": "diagnosis",
            },
            "depression": {
                "code": "F32",
                "description": "Depressive Episode",
                "category": "diagnosis",
            },
            "angina": {
                "code": "I20",
                "description": "Angina pectoris",
                "category": "diagnosis",
            },
            "asthma": {
                "code": "J45",
                "description": "Asthma bronchiale",
                "category": "diagnosis",
            },
            # Body parts/anatomy
            "kopf": {
                "code": "S01",
                "description": "Verletzung des Kopfes",
                "category": "anatomy",
            },
            "rücken": {
                "code": "M54",
                "description": "Rückenschmerzen",
                "category": "anatomy",
            },
            "herz": {
                "code": "I25",
                "description": "Chronische ischämische Herzkrankheit",
                "category": "anatomy",
            },
            "lunge": {
                "code": "J44",
                "description": "Sonstige chronische obstruktive Lungenkrankheit",
                "category": "anatomy",
            },
            "magen": {
                "code": "K25",
                "description": "Ulcus ventriculi",
                "category": "anatomy",
            },
            # Medications
            "aspirin": {
                "code": "N02BA01",
                "description": "Acetylsalicylsäure",
                "category": "medication",
            },
            "paracetamol": {
                "code": "N02BE01",
                "description": "Paracetamol",
                "category": "medication",
            },
            "metformin": {
                "code": "A10BA02",
                "description": "Metformin",
                "category": "medication",
            },
            "lisinopril": {
                "code": "C09AA03",
                "description": "Lisinopril",
                "category": "medication",
            },
            "ibuprofen": {
                "code": "M01AE01",
                "description": "Ibuprofen",
                "category": "medication",
            },
            "diclofenac": {
                "code": "M01AB05",
                "description": "Diclofenac",
                "category": "medication",
            },
            # Common rehab diagnoses (M-Kapitel = Muskel-Skelett)
            "rückenschmerzen": {
                "code": "M54",
                "description": "Rückenschmerzen/Dorsalgie",
                "category": "diagnosis",
            },
            "ischias": {
                "code": "M54",
                "description": "Rückenschmerzen/Ischialgie",
                "category": "diagnosis",
            },
            "hws-syndrom": {
                "code": "M53",
                "description": "Zervikalsyndrom",
                "category": "diagnosis",
            },
            "bws-syndrom": {
                "code": "M53",
                "description": "Thorakalsyndrom",
                "category": "diagnosis",
            },
            "lws-syndrom": {
                "code": "M53",
                "description": "Lumbalsyndrom",
                "category": "diagnosis",
            },
            "gonarthrose": {
                "code": "M17",
                "description": "Gonarthrose (Kniearthrose)",
                "category": "diagnosis",
            },
            "coxarthrose": {
                "code": "M16",
                "description": "Coxarthrose (Hüftarthrose)",
                "category": "diagnosis",
            },
            "tendinitis": {
                "code": "M77",
                "description": "Sonstige Enthesiopathien/Tendinitis",
                "category": "diagnosis",
            },
            "bandscheibenvorfall": {
                "code": "M51",
                "description": "Bandscheibenschäden",
                "category": "diagnosis",
            },
        }

    def lookup_entity(self, text: str) -> Optional[Dict[str, str]]:
        """Look up entity in ICD dictionary."""
        text_lower = text.lower().strip()

        # Direct lookup
        if text_lower in self.icd_codes:
            return self.icd_codes[text_lower]

        # Fuzzy matching for common variations
        for key, value in self.icd_codes.items():
            if self._fuzzy_match(text_lower, key):
                return value

        return None

    def _fuzzy_match(self, text: str, key: str) -> bool:
        """Simple fuzzy matching for medical terms."""
        # Check if key is contained in text or vice versa
        if key in text or text in key:
            return True

        # Check for common medical term variations
        variations = {
            "kopfschmerzen": ["kopfschmerz", "kopfweh", "kopfschmerz"],
            "schwindel": ["schwindelig", "schwindelgefühl"],
            "müdigkeit": ["müde", "erschöpfung", "ermüdung"],
            "fieber": ["temperatur", "fiebrig"],
            "schmerzen": ["schmerz", "weh", "wehweh"],
            "übelkeit": ["übel", "brechreiz"],
        }

        if key in variations:
            for variation in variations[key]:
                if variation in text:
                    return True

        return False


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    entities: List[MedicalEntity]
    processing_time: float
    model_used: List[str]
    text_length: int
    timestamp: datetime


class MedicalNERService:
    """Main service class for medical NER processing."""

    def __init__(self, config: ServiceConfig, model_loader: ModelLoader):
        self.config = config
        self.model_loader = model_loader
        self.icd_dict = ICDDictionary()
        self.icd_cache: Dict[str, Dict[str, str]] = {}
        self.extraction_stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "avg_processing_time": 0.0,
        }

        # Medical entity patterns for custom rules
        self._setup_medical_patterns()

    def _setup_medical_patterns(self):
        """Setup custom medical entity patterns."""
        self.medical_patterns = {
            # Drug patterns
            "MED_DRUG": [
                r"\b(?:Aspirin|Ibuprofen|Paracetamol|Metformin|Insulin)\b",
                r"\b[A-Z][a-z]+(?:in|ol|ide|ine)\b",  # Common drug suffixes
            ],
            # Disease patterns
            "DISEASE": [
                r"\b(?:Diabetes|Hypertension|Pneumonia|Bronchitis)\b",
                r"\b[A-Z][a-z]+itis\b",  # -itis diseases
            ],
            # Anatomy patterns
            "ANATOMY": [
                r"\b(?:Herz|Lunge|Leber|Niere|Gehirn)\b",
                r"\b[A-Z][a-z]+organ\b",
            ],
            # Symptoms
            "SYMPTOM": [
                r"\b(?:Schmerz|Fieber|Husten|Kopfschmerz)\b",
                r"\b[A-Z][a-z]+schmerz\b",
            ],
        }

    def _add_medical_patterns(self, nlp):
        """Add custom patterns for medical entities to spaCy pipeline."""
        matcher = Matcher(nlp.vocab)

        # Patterns for common medical terms
        patterns = [
            # Symptoms
            [{"LOWER": {"IN": ["kopfschmerzen", "kopfschmerz", "kopfweh"]}}],
            [{"LOWER": {"IN": ["schwindel", "schwindelig", "schwindelgefühl"]}}],
            [{"LOWER": {"IN": ["müdigkeit", "müde", "erschöpfung"]}}],
            [{"LOWER": {"IN": ["fieber", "temperatur", "fiebrig"]}}],
            [{"LOWER": {"IN": ["schmerzen", "schmerz", "weh"]}}],
            [{"LOWER": {"IN": ["übelkeit", "übel", "brechreiz"]}}],
            # Diagnoses
            [{"LOWER": {"IN": ["hypertension", "hochdruck", "bluthochdruck"]}}],
            [{"LOWER": {"IN": ["diabetes", "zuckerkrankheit", "diabetes mellitus"]}}],
            [{"LOWER": {"IN": ["depression", "depressiv", "deprimiert"]}}],
            [{"LOWER": {"IN": ["angina", "angina pectoris", "herzschmerz"]}}],
            [{"LOWER": {"IN": ["asthma", "asthma bronchiale", "atemnot"]}}],
            # Body parts
            [{"LOWER": {"IN": ["kopf", "haupt", "schädel"]}}],
            [{"LOWER": {"IN": ["rücken", "wirbelsäule", "spine"]}}],
            [{"LOWER": {"IN": ["herz", "kardial", "kardio"]}}],
            [{"LOWER": {"IN": ["lunge", "pulmonal", "respiratorisch"]}}],
            [{"LOWER": {"IN": ["magen", "gastro", "gastrointestinal"]}}],
            # Medications
            [{"LOWER": {"IN": ["aspirin", "acetylsalicylsäure", "asa"]}}],
            [{"LOWER": {"IN": ["paracetamol", "acetaminophen"]}}],
            [{"LOWER": {"IN": ["metformin", "glucophage"]}}],
            [{"LOWER": {"IN": ["lisinopril", "ace-hemmer"]}}],
        ]

        # Add patterns to matcher
        for i, pattern in enumerate(patterns):
            matcher.add(f"MEDICAL_TERM_{i}", [pattern])

        # Define a pipeline component that uses this matcher
        @Language.component("medical_matcher")
        def medical_matcher_component(doc):
            matches = matcher(doc)
            spans = [doc[start:end] for _, start, end in matches]
            doc.spans["medical_terms"] = spans
            return doc

        # Register it safely (no config injection)
        nlp.add_pipe("medical_matcher", last=True)

    def _add_physio_ruler(self, nlp):
        """Add an EntityRuler with common physiotherapy/rehab terms & abbreviations."""
        ruler: EntityRuler = nlp.add_pipe(
            "entity_ruler", name="physio_ruler", config={"overwrite_ents": False}
        )

        # Terms aligned with typical outpatient/inpatient rehab documentation
        physio_terms = [
            # treatments
            ("Krankengymnastik", "PHYSIO_TREATMENT"),
            ("KG", "PHYSIO_TREATMENT"),
            ("Manuelle Therapie", "PHYSIO_TREATMENT"),
            ("MT", "PHYSIO_TREATMENT"),
            ("Krankengymnastik am Gerät", "PHYSIO_TREATMENT"),
            ("KGG", "PHYSIO_TREATMENT"),
            ("Manuelle Lymphdrainage", "PHYSIO_TREATMENT"),
            ("MLD", "PHYSIO_TREATMENT"),
            ("Atemtherapie", "PHYSIO_TREATMENT"),
            ("Gangschule", "PHYSIO_TREATMENT"),
            ("Gangschulung", "PHYSIO_TREATMENT"),
            ("Triggerpunktbehandlung", "PHYSIO_TREATMENT"),
            ("Fango", "PHYSIO_TREATMENT"),
            ("Wärmetherapie", "PHYSIO_TREATMENT"),
            ("Kryotherapie", "PHYSIO_TREATMENT"),
            ("Elektrotherapie", "PHYSIO_TREATMENT"),
            ("TENS", "PHYSIO_TREATMENT"),
            ("Dehnung", "PHYSIO_TREATMENT"),
            ("Kräftigung", "PHYSIO_TREATMENT"),
            ("Gelenkmobilisation", "PHYSIO_TREATMENT"),
            # anatomy/regions shorthand often seen in notes
            ("HWS", "ANATOMY"),
            ("BWS", "ANATOMY"),
            ("LWS", "ANATOMY"),
            ("ISG", "ANATOMY"),
            ("Schulter", "ANATOMY"),
            ("Knie", "ANATOMY"),
            ("Hüfte", "ANATOMY"),
            ("Sprunggelenk", "ANATOMY"),
            # common complaints/goals
            ("Beweglichkeit", "FUNCTION"),
            ("ROM", "FUNCTION"),
            ("Belastbarkeit", "FUNCTION"),
            ("Gangbild", "FUNCTION"),
            ("Schmerzreduktion", "GOAL"),
            ("Aufbau", "GOAL"),
            ("Stabilisation", "GOAL"),
        ]

        patterns = [{"label": label, "pattern": term} for (term, label) in physio_terms]
        ruler.add_patterns(patterns)

    async def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract medical entities from a single text.

        Args:
            text: Input text to process

        Returns:
            ExtractionResult with detected entities
        """
        start_time = datetime.now()

        # Validate input
        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                processing_time=0.0,
                model_used=[],
                text_length=0,
                timestamp=start_time,
            )

        # Check text length limit
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]
            logger.warning(
                f"Text truncated to {self.config.max_text_length} characters"
            )

        try:
            # Extract entities using available models
            entities = []
            models_used = []

            # spaCy extraction (always available)
            if self.model_loader.is_model_loaded("spacy"):
                spacy_entities = await self._extract_with_spacy(text)
                entities.extend(spacy_entities)
                models_used.append("spacy")

            # GERNERMEDpp extraction
            if self.model_loader.is_model_loaded("gernermed"):
                gernermed_entities = await self._extract_with_gernermed(text)
                entities.extend(gernermed_entities)
                models_used.append("gernermed")

            # GermanBERT extraction
            if self.model_loader.is_model_loaded("germanbert"):
                germanbert_entities = await self._extract_with_germanbert(text)
                entities.extend(germanbert_entities)
                models_used.append("germanbert")

            # Apply custom medical patterns
            custom_entities = self._extract_with_patterns(text)
            entities.extend(custom_entities)

            # Merge and deduplicate entities
            merged_entities = self._merge_entities(entities)

            # Normalize entities
            normalized_entities = await self._normalize_entities(merged_entities)

            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(processing_time, len(normalized_entities))

            return ExtractionResult(
                entities=normalized_entities,
                processing_time=processing_time,
                model_used=models_used,
                text_length=len(text),
                timestamp=start_time,
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise

    async def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """
        Extract entities from multiple texts in batch.

        Args:
            texts: List of texts to process

        Returns:
            List of ExtractionResult objects
        """
        if len(texts) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {self.config.max_batch_size}"
            )

        # Process texts in parallel
        tasks = [self.extract_entities(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for text {i}: {result}")
                processed_results.append(
                    ExtractionResult(
                        entities=[],
                        processing_time=0.0,
                        model_used=[],
                        text_length=len(texts[i]),
                        timestamp=datetime.now(),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _extract_with_spacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using spaCy model."""
        try:
            nlp = self.model_loader.get_model("spacy")
            if not nlp:
                return []

            # Add custom patterns and physio ruler if not already added
            if "matcher" not in nlp.pipe_names:
                self._add_medical_patterns(nlp)
            if "physio_ruler" not in nlp.pipe_names:
                self._add_physio_ruler(nlp)

            doc = nlp(text)
            entities = []

            for ent in doc.ents:
                # Map spaCy labels to medical categories
                medical_label = self._map_spacy_label(ent.label_)

                # Look up ICD code
                icd_info = self.icd_dict.lookup_entity(ent.text)

                entity = MedicalEntity(
                    text=ent.text,
                    label=medical_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    source_model="spacy",
                    icd_code=icd_info.get("code") if icd_info else None,
                    icd_description=icd_info.get("description") if icd_info else None,
                    category=icd_info.get("category") if icd_info else None,
                    normalized_text=ent.text.lower(),
                )
                entities.append(entity)

            return entities

        except Exception as e:
            logger.error(f"spaCy extraction failed: {e}")
            return []

    async def _extract_with_gernermed(self, text: str) -> List[MedicalEntity]:
        """Extract entities using GERNERMEDpp model."""
        try:
            nlp = self.model_loader.get_model("gernermed")
            if not nlp:
                logger.warning("GERNERMEDpp model not loaded.")
                return []

            doc = nlp(text)
            entities: List[MedicalEntity] = []

            for ent in doc.ents:
                # Map the label from model to your internal labels
                mapped_label = self._map_spacy_label(ent.label_)
                icd_info = self.icd_dict.lookup_entity(ent.text)

                entity = MedicalEntity(
                    text=ent.text,
                    label=mapped_label,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=getattr(ent, "score", None) or 0.9,
                    source_model="gernermed",
                    normalized_text=ent.text.lower(),
                    icd_code=icd_info.get("code") if icd_info else None,
                    icd_description=icd_info.get("description") if icd_info else None,
                    category=icd_info.get("category") if icd_info else None,
                )
                entities.append(entity)

            logger.info(f"GERNERMEDpp extracted {len(entities)} entities.")
            return entities

        except Exception as e:
            logger.error(f"GERNERMEDpp extraction failed: {e}", exc_info=True)
            return []

    async def _extract_with_germanbert(self, text: str) -> List[MedicalEntity]:
        """Extract entities using GermanBERT model."""
        try:
            # Placeholder for GermanBERT integration
            # This would be implemented based on the specific model's API
            logger.info("GermanBERT extraction not yet implemented")
            return []

        except Exception as e:
            logger.error(f"GermanBERT extraction failed: {e}")
            return []

    def _extract_with_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract entities using custom medical patterns."""
        entities = []

        for label, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = MedicalEntity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,  # Pattern-based confidence
                        source_model="patterns",
                    )
                    entities.append(entity)

        return entities

    def _map_spacy_label(self, spacy_label: str) -> str:
        """
        Map spaCy labels to medAI-normalized categories.
        - From base model: PERSON/ORG/GPE/DATE/etc.
        - From GERNERMED: Drug/Strength/Route/Form/Dosage/Frequency/Duration
        - From physio ruler: PHYSIO_TREATMENT/ANATOMY/FUNCTION/GOAL
        """
        base_map = {
            # generic
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "PERCENT": "MEASUREMENT",
            "MONEY": "MEASUREMENT",
            "QUANTITY": "MEASUREMENT",
            "CARDINAL": "NUMBER",
            "ORDINAL": "NUMBER",
            # physio ruler
            "PHYSIO_TREATMENT": "PHYSIO_TREATMENT",
            "ANATOMY": "ANATOMY",
            "FUNCTION": "FUNCTION",
            "GOAL": "GOAL",
            # GERNERMED (med7-style)
            "Drug": "MED_DRUG",
            "Strength": "MED_STRENGTH",
            "Route": "MED_ROUTE",
            "Form": "MED_FORM",
            "Dosage": "MED_DOSAGE",
            "Frequency": "MED_FREQUENCY",
            "Duration": "MED_DURATION",
        }
        return base_map.get(spacy_label, "MISC")

    def _merge_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Merge overlapping entities and remove duplicates."""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: x.start)

        merged = []
        current = entities[0]

        for entity in entities[1:]:
            # Check for overlap
            if entity.start < current.end and entity.end <= current.end:
                # Entity is contained within current, skip
                continue
            elif entity.start < current.end:
                # Overlapping entities, choose the one with higher confidence
                if entity.confidence > current.confidence:
                    current = entity
            else:
                # No overlap, add current and move to next
                merged.append(current)
                current = entity

        merged.append(current)
        return merged

    async def _normalize_entities(
        self, entities: List[MedicalEntity]
    ) -> List[MedicalEntity]:
        """Normalize entity text and add ICD codes."""
        normalized = []

        for entity in entities:
            # Normalize text
            normalized_text = self._normalize_text(entity.text)
            entity.normalized_text = normalized_text

            # Look up ICD code if enabled
            if ICD_CONFIG["lookup_enabled"]:
                icd_info = await self._lookup_icd_code(normalized_text, entity.label)
                if icd_info:
                    entity.icd_code = icd_info.get("code")
                    entity.icd_description = icd_info.get("description")

            normalized.append(entity)

        return normalized

    def _normalize_text(self, text: str) -> str:
        """Normalize entity text."""
        # Basic normalization
        normalized = text.strip()
        normalized = re.sub(r"\s+", " ", normalized)  # Normalize whitespace
        normalized = normalized.title()  # Title case

        return normalized

    async def _lookup_icd_code(self, text: str, label: str) -> Optional[Dict[str, str]]:
        """Look up ICD code for entity text."""
        # Check cache first
        cache_key = f"{text}_{label}"
        if cache_key in self.icd_cache:
            return self.icd_cache[cache_key]

        # Placeholder for ICD lookup
        # This would integrate with an actual ICD database
        icd_info = None

        # Cache the result
        if icd_info:
            self.icd_cache[cache_key] = icd_info
            # Limit cache size
            if len(self.icd_cache) > ICD_CONFIG["cache_size"]:
                # Remove oldest entries
                keys_to_remove = list(self.icd_cache.keys())[
                    : len(self.icd_cache) - ICD_CONFIG["cache_size"]
                ]
                for key in keys_to_remove:
                    del self.icd_cache[key]

        return icd_info

    def _update_stats(self, processing_time: float, entity_count: int):
        """Update extraction statistics."""
        self.extraction_stats["total_extractions"] += 1
        self.extraction_stats["total_entities"] += entity_count

        # Update average processing time
        total_extractions = self.extraction_stats["total_extractions"]
        current_avg = self.extraction_stats["avg_processing_time"]
        self.extraction_stats["avg_processing_time"] = (
            current_avg * (total_extractions - 1) + processing_time
        ) / total_extractions

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return {
            "extraction_stats": self.extraction_stats.copy(),
            "model_status": self.model_loader.get_model_status(),
            "config": {
                "max_text_length": self.config.max_text_length,
                "max_batch_size": self.config.max_batch_size,
                "models_enabled": {
                    "spacy": self.config.spacy_enabled,
                    "gernermed": self.config.gernermed_enabled,
                    "germanbert": self.config.germanbert_enabled,
                },
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": self.model_loader.get_model_status(),
            "stats": self.extraction_stats,
            "memory_usage": self._get_memory_usage(),
        }

        # Check if at least one model is loaded
        if not any(self.model_loader.get_model_status().values()):
            health_status["status"] = "unhealthy"
            health_status["error"] = "No models loaded"

        return health_status

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    def get_entity_statistics(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        """Get statistics for extracted entities."""
        stats = {
            "total_entities": len(entities),
            "by_category": {},
            "by_label": {},
            "with_icd_codes": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }
        for e in entities:
            cat = e.category or "unknown"
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            stats["by_label"][e.label] = stats["by_label"].get(e.label, 0) + 1
            if e.icd_code:
                stats["with_icd_codes"] += 1
            if e.confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif e.confidence > 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
        return stats

    def format_entities_for_display(self, entities: List[MedicalEntity]) -> str:
        """Format entities for display."""
        if not entities:
            return "Keine medizinischen Entitäten gefunden."
        lines = []
        for e in entities:
            line = f"• {e.text} ({e.label})"
            if e.icd_code:
                line += f" – ICD: {e.icd_code}"
            if e.icd_description:
                line += f" – {e.icd_description}"
            lines.append(line)
        return "\n".join(lines)
