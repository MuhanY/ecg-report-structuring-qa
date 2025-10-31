# ecg_qa_diagnose_validate_v5.py
# Streamlined version that leverages subcategory names as natural rollups
# Only needs minimal cross-subcategory rollup definitions

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import random
import uuid
import json
import itertools
import collections

# --------------------------
# Config
# --------------------------

DIAGNOSE_VALIDATE_ENTITY_PRESENCE = False
DIAGNOSE_VALIDATE_ATTRIBUTES = False
REASONING_QUESTIONS_ENABLED = True

SCHEMA_CONFIG = {
    # Categories to include in presence questions
    "presence_whitelist": {
        "Impression": None,
        "Rhythm": None,
        "Extranodal_conduction_abnormalities": None,
        "Chambers": None,
        "Primary_findings": None,
        "Axis_and_voltage": None,
        "Pacemaker": None,
    },
    
    # ========================================================================
    # MANUAL CATEGORY SCOPE OVERRIDES
    # ========================================================================
    # Force specific subcategories to use category scope
    # Use this for special cases not caught by suffix rules
    # ========================================================================
    
    "force_category_scope_for_subcategory": {
        "Other_findings",           # Awkward 
        "other_impressions",         # Generic bucket
        "overall_intepretation",     # Too generic
        
        "sinus_rhythm",
        "myocardial_injury",        
        "Q_wave_abnormalities",
        "S_wave_abnormalities",
        "U_wave_abnormalities",
        "PR_segment_abnormalities",
        "RR_interval_abnormalities"
    },

    # ========================================================================
    # CONCEPT-LEVEL SCOPE OVERRIDES (for umbrella terms in reports)
    # ========================================================================
    # Specific concepts that should use category scope even if subcategory doesn't
    # Use this for cross-subcategory umbrella terms from your reports
    # ========================================================================
    
    "use_category_scope": {
        # Cross-subcategory umbrella terms
        "ST-T wave abnormalities": True,
        "myocardial_ischemia": True,
        "Abnormal_R_wave_progression": True,
        
        # Single-subcateory umbrella terms (not subcategory names)
        "Bundle_branch_block": True,
        "Fascicular_block": True,
    },

    # ========================================================================
    # MINIMAL ROLLUP DEFINITIONS (cross-subcategory only)
    # ========================================================================
    # Only define rollups for umbrella terms that span multiple subcategories
    # Subcategory-level rollups are handled automatically
    # ========================================================================
    
    "rollups": {
        # Cross-subcategory umbrella (spans ST and T wave subcategories)
        "ST-T wave abnormalities": [
            "ST segment elevation", "ST segment depression",
            "T wave inversion", "Tall T wave", "Flat T wave", "Wide T wave"
        ],
        
        # Cross-category rollup
        "myocardial_ischemia": ["myocardial_infarction"],
        
        # Generic R wave progression
        "Abnormal_R_wave_progression": [
            "Abnormal precordial R wave progression",
            "Delayed precordial R wave progression",
            "Early precordial R wave progression"
        ],
        "Abnormal precordial R wave progression": [
            "Delayed precordial R wave progression",
            "Early precordial R wave progression"
        ],
        
        # Electrolyte abnormalities
        "electrolyte_abnormality": ["hyperkalemia", "hypokalemia"],
        
        # BBB and fascicular blocks
        "Bundle_branch_block": [
            "Left bundle-branch block",
            "Right bundle-branch block"
        ],
        "Fascicular_block": [
            "Left anterior fascicular block",
            "Left posterior fascicular block"
        ],
    },

    # ========================================================================
    # OTHER CONFIGURATIONS
    # ========================================================================

    # Attribute fields for attribute questions
    "attribute_fields": {
        "perceptual": ["Location_lead", "Location_signal", "Quality (severity)", "Type", "Conduction", 
                       "Modifier", "Completeness", "Degree", "Focus", "Temporality", "Frequency", 
                       "Rate_response", "Morphology", "Acuity", "Chamber", "Sensing", "Pacing", 
                       "Criteria", "Type (depth)"],
        "topological": ["Type (location)", "Reciprocality"],
        "epistemic": ["Uncertainty", "Quality (specificity)"],
    },

    # Number of options
    "num_options": {
        "presence_single": 4,
        "presence_multi": 5,
        "attribute_single": 4,
        "attribute_multi": 5,
        "evidence_attribution_single": 4,
        "evidence_attribution_multi": 5,
        "inference_single": 4,
        "inference_multi": 5,
    },

    # Multi-label limits
    "multilabel_caps": {
        "presence": (1, 3),
        "attribute": (1, 4),
        "evidence_attribution": (1, 3),
        "inference": (1, 2),
    },

    # Minimal pool sizes
    "min_pool_sizes": {
        "presence": 3,
        "attribute": 2,
        "evidence_attribution": 2,
        "inference": 2,
    },

    # Attribute pretty names
    "attr_pretty": {
        "Location_lead": "lead(s)",
        "Location_signal": "location",
        "Morphology": "morphology",
        "Acuity": "acuity",
        "Type": "type",
        "Type (location)": "location",
        "Type (depth)": "depth",
        "Quality (severity)": "severity",
        "Uncertainty": "certainty level",
        "Quality (specificity)": "specificity",
        "Sensing": "sensing mode",
        "Pacing": "pacing mode",
        "Criteria": "diagnostic criteria",
        "Chamber": "chamber",
        "Conduction": "conduction pattern",
        "Frequency": "frequency",
        "Rate_response": "rate response",
        "Focus": "focus",
        "Modifier": "modifier",
    },

    # Category pretty names
    "category_pretty": {
        "Primary_findings": "ECG findings",
        "Impression": "diagnoses",
        "Rhythm": "rhythm",
        "Chambers": "chamber abnormalities",
        "Axis_and_voltage": "axis or voltage abnormalities",
        "Pacemaker": "pacemaker findings",
        "General_description": "technical or descriptive findings",
    },

    # Subcategory pretty names
    "subcategory_pretty": {
        # Rhythm
        "sinus_rhythm": "rhythm findings",
        "sinus_node_arrhythmias": "sinus node arrhythmias",
        "junctional_rhythm": "junctional rhythms",
        "ectopic_atrial_rhythm": "ectopic atrial rhythms",
        "supraventricular_arrhythmias": "supraventricular arrhythmias",
        "supraventricular_tachyarrhythmias": "supraventricular tachyarrhythmias",
        "ventricular_arrhythmias": "ventricular arrhythmias",
        "ventricular_tachyarrhythmias": "ventricular tachyarrhythmias",
        "atrioventricular_conduction_abnormalities": "AV conduction abnormalities",
        "intraventricular_conduction_abnormalities": "intraventricular conduction abnormalities",
        "intraatrial_conduction_abnormalities": "intraatrial conduction abnormalities",
        
        # Extranodal conduction abnormalities
        "Extranodal_conduction_abnormalities": "extranodal conduction abnormalities",
        
        # Primary findings (with _abnormalities suffix - will use category scope automatically)
        "P_wave_abnormalities": "P wave abnormalities",
        "Q_wave_abnormalities": "Abnormal Q wave",
        "R_wave_abnormalities": "R wave abnormalities",
        "S_wave_abnormalities": "Abnormal S wave",
        "T_wave_abnormalities": "T wave abnormalities",
        "U_wave_abnormalities": "Abnormal U wave",
        "QRS_complex_abnormalities": "QRS complex abnormalities",
        "ST_segment_abnormalities": "ST segment abnormalities",
        "PR_interval_abnormalities": "PR interval abnormalities",
        "PR_segment_abnormalities": "PR segment abnormalities",
        "QT_interval_abnormalities": "QT interval abnormalities",
        "RR_interval_abnormalities": "RR interval abnormalities",
        "Repolarization_abnormalities": "repolarization abnormalities",
        "Other_findings": "other findings",
        
        # Axis and voltage
        "qrs_axis_abnormalities": "Abnormal QRS axis",
        "p_axis_abnormalities": "Abnormal P axis",
        "voltage_abnormalities": "voltage abnormalities",
        
        # Chambers
        "atrial_enlargement": "atrial enlargement",
        "ventricular_hypertrophy": "ventricular hypertrophy",
        
        # Pacemaker
        "pacemaker_rhythm": "pacemaker rhythms",
        "pacemaker_failure": "pacemaker malfunctions",
        
        # Impression
        "myocardial_injury": "myocardial injury",
        "other_impressions": "clinical impressions",
        
        # General
        "overall_intepretation": "overall interpretation",
        "technical_condition": "technical conditions",
    },

    # Random seed
    "seed": 1337,
}


@dataclass
class EntityItem:
    entity_id: str
    category: str
    subcategory: Optional[str]
    concept: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_raw(raw: Dict[str, Any]) -> "EntityItem":
        e = raw.get("entity", {}) if "entity" in raw else raw
        attrs = dict(raw.get("attributes", {}))

        # Extract nested attribute values
        for k, v in attrs.items():
            if isinstance(v, dict) and "value" in v:
                attrs[k] = v["value"]
        
        # # Include other top-level fields as attributes (like occurrences_count)
        # for k, v in raw.items():
        #     if k not in ("id", "entity") and not isinstance(v, (dict, list)):
        #         attrs.setdefault(k, v)

        return EntityItem(
            entity_id=raw.get("id", str(uuid.uuid4())),
            category=e.get("category", "Unknown"),
            subcategory=e.get("subcategory"),
            concept=e.get("concept", "Unknown"),
            attributes=attrs,
        )


@dataclass
class QAPool:
    # concept_pool: Dict[Tuple[str, Optional[str]], Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    # category_pool: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))
    attribute_pool: Dict[str, Set[Any]] = field(default_factory=lambda: collections.defaultdict(set))
    
    schema_concept_pool: Dict[Tuple[str, Optional[str]], Set[str]] = field(default_factory=lambda: collections.defaultdict(set))  # (category, subcategory) -> concepts
    schema_category_pool: Dict[str, Set[str]] = field(default_factory=lambda: collections.defaultdict(set))  # category -> concepts
    schema_attribute_pool: Dict[str, Set[Any]] = field(default_factory=lambda: collections.defaultdict(set))  # attribute name -> values

    schema_subcategory_list: Set[str] = field(default_factory=set)  # Set of all subcategories in schema

    def add_attributes_from_entity(self, ent: EntityItem):
        # key = (ent.category, ent.subcategory)
        # self.concept_pool[key].add(ent.concept)
        # self.category_pool[ent.category].add(ent.concept)
    
        for k, v in ent.attributes.items():
            if v is None:
                continue
            if isinstance(v, (list, tuple, set)):
                for x in v:
                    self.attribute_pool[k].add(x)
            else:
                self.attribute_pool[k].add(v)

    def load_schema(self, schema_path: str):
        """Load schema to build comprehensive distractor pools"""
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Track all subcategories in schema
        for entity_group in schema.get("entities_schema", []):
            for subcat_name, subcat_data in entity_group.get("subcategories", {}).items():
                self.schema_subcategory_list.add(self._normalize_concept(subcat_name))

        # Build concept and attribute pools from schema
        for entity_group in schema.get("entities_schema", []):
            category = entity_group.get("category")
            for subcat_name, subcat_data in entity_group.get("subcategories", {}).items():
                concepts = subcat_data.get("concepts", [])  # List of concepts in this subcategory
                concepts = [c for c in concepts if self._normalize_concept(c) not in self.schema_subcategory_list]  # Filter out subcategory-based umbrella concepts
                # print(concepts)
                key = (category, subcat_name)
                self.schema_concept_pool[key].update(concepts)  # Add concepts to (category, subcategory) pool
                self.schema_category_pool[category].update(concepts)  # Add concepts to category pool
                
                attr_hints = subcat_data.get("attribute_hints_add", {})  # Attribute hints at subcategory level
                for attr_name, values in attr_hints.items():  # Add attribute hints to attribute pool
                    self.schema_attribute_pool[attr_name].update(values)
                
                for group in subcat_data.get("groups", []):  # Process groups within subcategory
                    # group_concepts = group.get("concepts", [])  # Concepts in this group
                    # self.schema_concept_pool[key].update(group_concepts)
                    # self.schema_category_pool[category].update(group_concepts)
                    
                    group_hints = group.get("attribute_hints_add", {})  # Attribute hints at group level
                    for attr_name, values in group_hints.items():  # Add group attribute hints to attribute pool
                        self.schema_attribute_pool[attr_name].update(values)

        defaults = schema.get("defaults", {})
        default_hints = defaults.get("attribute_hints", {})
        for attr_name, values in default_hints.items():
            self.schema_attribute_pool[attr_name].update(values)

    def get_concept_pool(self, category: str, subcategory: Optional[str], use_category_scope: bool = False) -> Set[str]:
        """Get combined concept pool with optional category-wide scope"""
        if use_category_scope:
            schema_pool = self.schema_category_pool.get(category, set())
        else:
            key = (category, subcategory)
            schema_pool = self.schema_concept_pool.get(key, set())
                        
        return schema_pool

    def get_attribute_pool(self, attribute: str) -> Set[Any]:
        """Get combined attribute pool (corpus + schema)"""
        corpus_pool = self.attribute_pool.get(attribute, set())
        schema_pool = self.schema_attribute_pool.get(attribute, set())
        return corpus_pool | schema_pool


    @staticmethod
    def _normalize_concept(concept: str) -> str:
        """Normalize concept name for rollup matching"""
        return concept.strip().lower().replace(" ", "_").replace("-", "_")
    
class DiagnoseValidateQAGenerator:
    def __init__(self, config: Dict[str, Any] = None, schema_path: Optional[str] = None):
        self.cfg = config or SCHEMA_CONFIG
        random.seed(self.cfg["seed"])
        self._pool = QAPool()
        
        self._build_rollup_index()
        
        if schema_path:
            self._pool.load_schema(schema_path)

    def _build_rollup_index(self):
        """Build bidirectional index of rollup relationships"""
        self.rollup_children = {} # key: parent, value: list of children
        self.rollup_parents = {} # key: child, value: list of parents
        
        for parent, children in self.cfg["rollups"].items():
            parent_norm = self._normalize_concept(parent)
            self.rollup_children[parent_norm] = [self._normalize_concept(c) for c in children]
            
            for child in children:
                child_norm = self._normalize_concept(child)
                if child_norm not in self.rollup_parents:
                    self.rollup_parents[child_norm] = []
                self.rollup_parents[child_norm].append(parent_norm)

    @staticmethod
    def _normalize_concept(concept: str) -> str:
        """Normalize concept name for rollup matching"""
        return concept.strip().lower().replace(" ", "_").replace("-", "_")

    def _get_rollup_related(self, concept: str) -> Set[str]:
        """Get all concepts related via rollup hierarchy"""
        norm = self._normalize_concept(concept)
        related = set()
        
        if norm in self.rollup_children:  # if norm is a parent, get children
            related.update(self.rollup_children[norm])
            
            for child in list(related):  # if get grandchildren.
                if child in self.rollup_children:
                    related.update(self.rollup_children[child])
        
        if norm in self.rollup_parents:  # if norm is a child, get parents
            related.update(self.rollup_parents[norm])

        return related  # return parents, children, and grandchildren set

    def _should_use_category_scope(self, concept: str, subcategory: Optional[str]) -> bool:
        """Determine if this concept should use category scope"""
        # Check concept-specific override first
        if concept in self.cfg["use_category_scope"]:
            return self.cfg["use_category_scope"][concept]
        
        # Check subcategory-based umbrella concepts
        if self._normalize_concept(concept) == self._normalize_concept(subcategory) and subcategory is not None:
            return True
        
        # Check if subcategory is in manual override list
        if subcategory in self.cfg["force_category_scope_for_subcategory"]:
            return True
        
        # Check if it's a rollup parent (has children)
        norm = self._normalize_concept(concept)
        if norm in self.rollup_children:
            return True
        
        return False

    ### Public API methods
    # Build reports-based attribute pool from list of reports
    def build_attribute_pool(self, reports: List[Dict[str, Any]]):
        for r in reports:
            for ent_raw in r.get("entities", []):
                ent = EntityItem.from_raw(ent_raw)
                self._pool.add_attributes_from_entity(ent)

    # Generate QA items for a list of reports
    def generate_for_reports(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.build_attribute_pool(reports)
        qa_items: List[Dict[str, Any]] = []
        for r in reports:
            qa_items.extend(self.generate_for_report(r))
        return qa_items

    # Generate QA items for a single report
    def generate_for_report(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        entities = [EntityItem.from_raw(e) for e in report.get("entities", [])]  # Extract entities from report
        qa_items: List[Dict[str, Any]] = []

        by_cat = collections.defaultdict(list)  # key: (category, subcategory), value: list of entities
        for e in entities:
            by_cat[(e.category, e.subcategory)].append(e)

        # 1-1: Diagnose/Validate entity presence
        if DIAGNOSE_VALIDATE_ENTITY_PRESENCE:
            for (cat, subcat), ents in by_cat.items():
                if not self._is_presence_allowed(cat):  # Skip categories not in whitelist
                    continue

                use_cat_scope = any(self._should_use_category_scope(e.concept, subcat) for e in ents)  # Determine if category scope should be used
            
                pool = self._pool.get_concept_pool(cat, subcat, use_category_scope=use_cat_scope)  # Get distractor pool
                if len(pool) < self.cfg["min_pool_sizes"]["presence"]:  # Skip if pool too small
                    continue

                qa_items.append(self._make_presence_item(report, cat, subcat, ents, use_cat_scope))

        # 1-2: Diagnose/Validate attributes for entities
        if DIAGNOSE_VALIDATE_ATTRIBUTES:
            for ent in entities:
                for role, fields in self.cfg["attribute_fields"].items():
                    for attr in fields:
                        item = self._make_attribute_item(report, ent, attr, role)
                        if item:
                            qa_items.append(item)
                        
        # 2 Numeric quantification
        
        # 3: Reasoning
        if REASONING_QUESTIONS_ENABLED:
            relations = report.get("relations", [])
            due_to_rels = [r for r in relations if r.get("relation") == "Due_to"]
            
            if due_to_rels:
                # Build entity lookup
                ent_by_id = {e.entity_id: e for e in entities}
                
                # Group by tail (impression)
                by_impression = collections.defaultdict(list)
                for rel in due_to_rels:
                    tail_id = rel.get("tail")
                    head_id = rel.get("head")
                    if tail_id in ent_by_id and head_id in ent_by_id:
                        by_impression[tail_id].append(head_id)
                
                # 3-1: Evidence Attribution (Which findings justify this impression?)
                for impression_id, finding_ids in by_impression.items():
                    if len(finding_ids) > 0:
                        item = self._make_evidence_attribution_item(
                            report, ent_by_id[impression_id], 
                            [ent_by_id[fid] for fid in finding_ids], entities
                        )
                        if item:
                            qa_items.append(item)
                
                # 3-2: Inference (Which diagnosis is implied by these findings?)
                for impression_id, finding_ids in by_impression.items():
                    if len(finding_ids) > 0:
                        item = self._make_inference_item(
                            report, ent_by_id[impression_id],
                            [ent_by_id[fid] for fid in finding_ids], entities
                        )
                        if item:
                            qa_items.append(item)
        
                # 3-3 Counterfactual reasoning
        
        # 4 Confounders
        # 5 Recommendation
        
        # 6 Temporal dynamics

        return qa_items

    def _make_presence_item(self, report, category, subcategory, ents_in_report: List[EntityItem], 
                           use_category_scope: bool) -> Dict[str, Any]:
        concepts_present = sorted({e.concept for e in ents_in_report})  # Category-subcategory concepts in report
        pool = sorted(self._pool.get_concept_pool(category, subcategory, use_category_scope=use_category_scope))  # Distractor pool

        min_c, max_c = self.cfg["multilabel_caps"]["presence"]
        k_correct = min(max(len(concepts_present), 1), max_c)
        correct = sorted(random.sample(concepts_present, k=k_correct)) if len(concepts_present) > k_correct else concepts_present

        # Filter distractors using rollup hierarchy
        distractors = []
        correct_related = set()
        for c in correct:  # Gather all related concepts via rollups
            correct_related.update(self._get_rollup_related(c))
            correct_related.add(self._normalize_concept(c))
        
        for candidate in pool:  # Filter distractors by correct answers and their related concepts (parents/children/grandchildren)
            if candidate in concepts_present:
                continue
            if self._normalize_concept(candidate) in correct_related:
                continue
            distractors.append(candidate)
        
        random.shuffle(distractors)

        num_opts = self.cfg["num_options"]["presence_multi"] if len(correct) > 1 else self.cfg["num_options"]["presence_single"]
        needed = max(0, num_opts - len(correct))
        options = correct + distractors[:needed]
        random.shuffle(options)

        multi_label = len(correct) > 1
        
        # Build question
        if use_category_scope:
            entity_type = self.cfg["category_pretty"].get(category, category)
        else:
            entity_type = self.cfg["subcategory_pretty"].get(subcategory, subcategory)
        
        question = f"Which {entity_type} are present in this ECG?" if multi_label else f"Which {entity_type} is present in this ECG?"

        return {
            "qa_id": str(uuid.uuid4()),
            "report_id": report.get("report_id"),
            "family": "DiagnoseValidate",
            "question_type": "presence_multi" if multi_label else "presence_single",
            "difficulty": "D0",
            "question": question,
            "options": options,
            "answer": correct if multi_label else correct[0],
            "meta": {
                "scope_category": category,
                "scope_subcategory": subcategory,
                "scope_level": "category" if use_category_scope else "subcategory",
                "attribute_role": None,
            },
        }

    def _is_presence_allowed(self, category: str) -> bool:
        return category in self.cfg["presence_whitelist"]

    def _make_attribute_item(self, report: Dict[str, Any], ent: EntityItem, attr: str, role: str) -> Optional[Dict[str, Any]]:
        raw_v = ent.attributes.get(attr)
        if raw_v is None:
            return None

        if isinstance(raw_v, (list, tuple, set)):
            values = [self._norm_val(v) for v in raw_v if v is not None]
        else:
            values = [self._norm_val(raw_v)]

        if not values:
            return None

        pool = sorted({self._norm_val(v) for v in self._pool.get_attribute_pool(attr) if v is not None})
        
        if len(pool) < self.cfg["min_pool_sizes"]["attribute"]:
            return None

        min_c, max_c = self.cfg["multilabel_caps"]["attribute"]
        k_correct = min(max(len(values), 1), max_c)
        correct = sorted(random.sample(values, k=k_correct)) if len(values) > k_correct else sorted(values)
        multi_label = len(correct) > 1

        distractors = [x for x in pool if x not in correct]
        random.shuffle(distractors)
        num_opts = self.cfg["num_options"]["attribute_multi"] if multi_label else self.cfg["num_options"]["attribute_single"]
        needed = max(0, num_opts - len(correct))
        options = correct + distractors[:needed]
        random.shuffle(options)

        pretty_attr = self.cfg["attr_pretty"].get(attr, attr)
        # scope = self._format_scope(ent)
        
        if multi_label:
            q = f"Which {pretty_attr} apply to {ent.concept}?"
        else:
            q = f"Which is the {pretty_attr} for {ent.concept}?"

        return {
            "qa_id": str(uuid.uuid4()),
            "report_id": report.get("report_id"),
            "family": "DiagnoseValidate",
            "question_type": "attribute_multi" if multi_label else "attribute_single",
            "difficulty": self._difficulty_for_role(role),
            "question": q,
            "options": options,
            "answer": correct if multi_label else correct[0],
            "meta": {
                "entity_id": ent.entity_id,
                "entity_category": ent.category,
                "entity_subcategory": ent.subcategory,
                "entity_concept": ent.concept,
                "attribute": attr,
                "attribute_role": role,
                "scope_level": "concept (attribute)"
            },
        }

    # @staticmethod
    # def _format_scope(ent: EntityItem) -> str:
    #     cat = ent.category
    #     sub = f" / {ent.subcategory}" if ent.subcategory else ""
    #     return f"{ent.concept} ({cat}{sub})"

    def _make_evidence_attribution_item(self, report: Dict[str, Any], impression: EntityItem,
                                               findings: List[EntityItem], all_entities: List[EntityItem]) -> Optional[Dict[str, Any]]:
        """
        Create a question asking which findings justify/support an impression or diagnosis.
        Template: "Which findings suggest {impression}?"
        Distractors: findings from same subcategory/category, or other findings
        """
        if not findings:
            return None
        
        # Get correct findings (concepts)
        correct_findings = sorted({f.concept for f in findings})
        
        # Build distractor pool
        distractors = []
        
        # Strategy 1: Findings from same subcategory/category as the correct findings
        for f in findings:
            # Same subcategory
            same_subcat = [e.concept for e in all_entities 
                          if e.category == f.category and e.subcategory == f.subcategory 
                          and e.concept not in correct_findings]
            distractors.extend(same_subcat)
            
            # Same category
            same_cat = [e.concept for e in all_entities
                       if e.category == f.category and e.subcategory != f.subcategory
                       and e.concept not in correct_findings]
            distractors.extend(same_cat)
        
        # Strategy 2: Add other Primary_findings not related to this impression
        other_findings = [e.concept for e in all_entities
                         if e.category == "Primary_findings" and e.concept not in correct_findings]
        distractors.extend(other_findings)
        
        # Strategy 3: Add Axis_and_voltage findings as distractors
        axis_findings = [e.concept for e in all_entities
                        if e.category == "Axis_and_voltage" and e.concept not in correct_findings]
        distractors.extend(axis_findings)
        
        # Deduplicate and shuffle
        distractors = sorted(set(distractors))
        random.shuffle(distractors)
        
        # Check if we have enough options
        if len(distractors) < self.cfg["min_pool_sizes"]["evidence_attribution"]:
            return None
        
        # Determine multi-label
        min_c, max_c = self.cfg["multilabel_caps"]["evidence_attribution"]
        k_correct = min(max(len(correct_findings), 1), max_c)
        correct = sorted(random.sample(correct_findings, k=k_correct)) if len(correct_findings) > k_correct else correct_findings
        multi_label = len(correct) > 1
        
        # Build options
        num_opts = self.cfg["num_options"]["evidence_attribution_multi"] if multi_label else self.cfg["num_options"]["evidence_attribution_single"]
        needed = max(0, num_opts - len(correct))
        options = correct + distractors[:needed]
        random.shuffle(options)
        
        # Build question - adapt based on category
        if impression.category == "Impression":
            question_templates = [
                f"Which findings suggest {impression.concept}?",
                f"Which findings support the diagnosis of {impression.concept}?",
                f"Which ECG findings justify {impression.concept}?",
            ]
        else:
            question_templates = [
                f"Which findings suggest {impression.concept}?",
                f"Which ECG findings justify {impression.concept}?",
                f"Which findings are associated with {impression.concept}?",
            ]
        question = random.choice(question_templates)
        
        return {
            "qa_id": str(uuid.uuid4()),
            "report_id": report.get("report_id"),
            "family": "DiagnoseValidate",
            "question_type": "evidence_attribution_multi" if multi_label else "evidence_attribution_single",
            "difficulty": "D1",
            "question": question,
            "options": options,
            "answer": correct if multi_label else correct[0],
            "meta": {
                "impression_id": impression.entity_id,
                "impression_category": impression.category,
                "impression_subcategory": impression.subcategory,
                "impression_concept": impression.concept,
                "correct_finding_ids": [f.entity_id for f in findings if f.concept in correct],
                "scope_level": "relationship (evidence_attribution)",
            },
        }

    def _make_inference_item(self, report: Dict[str, Any], impression: EntityItem,
                                    findings: List[EntityItem], all_entities: List[EntityItem]) -> Optional[Dict[str, Any]]:
        """
        Create a question asking which diagnosis is implied by given findings.
        Template: "Which diagnosis is implied by {findings_set}?"
        Distractors: other impressions, or same impression with different attributes
        """
        if not findings:
            return None
        
        # Only create inference questions for Impression category
        if impression.category != "Impression":
            return None
        
        # Select findings to show (limit to 2-3 for readability)
        k_findings = min(len(findings), 3)
        selected_findings = random.sample(findings, k=k_findings)
        findings_text = ", ".join([f.concept for f in selected_findings])
        
        # Correct answer
        correct_concept = impression.concept
        
        # Build distractor pool: other impressions
        distractors = []
        
        # Strategy 1: Same subcategory (e.g., other myocardial injuries)
        same_subcat = [e.concept for e in all_entities
                      if e.category == "Impression" and e.subcategory == impression.subcategory
                      and e.concept != correct_concept]
        distractors.extend(same_subcat)
        
        # Strategy 2: Other impressions from different subcategories
        other_impressions = [e.concept for e in all_entities
                            if e.category == "Impression" and e.concept != correct_concept]
        distractors.extend(other_impressions)
        
        # Strategy 3: Get impressions from pool if not enough in report
        pool_impressions = self._pool.get_concept_pool("Impression", impression.subcategory, use_category_scope=True)
        distractors.extend([c for c in pool_impressions if c != correct_concept])
        
        # Deduplicate and shuffle
        distractors = sorted(set(distractors))
        random.shuffle(distractors)
        
        # Check if we have enough options
        if len(distractors) < self.cfg["min_pool_sizes"]["inference"]:
            return None
        
        # Build options (single answer for inference)
        correct = [correct_concept]
        multi_label = False
        
        num_opts = self.cfg["num_options"]["inference_single"]
        needed = max(0, num_opts - len(correct))
        options = correct + distractors[:needed]
        random.shuffle(options)
        
        # Build question
        question_templates = [
            f"Which diagnosis is implied by {findings_text}?",
            f"What diagnosis do these findings suggest: {findings_text}?",
            f"Based on {findings_text}, what is the most likely diagnosis?",
        ]
        question = random.choice(question_templates)
        
        return {
            "qa_id": str(uuid.uuid4()),
            "report_id": report.get("report_id"),
            "family": "DiagnoseValidate",
            "question_type": "inference_single",
            "difficulty": "D2",
            "question": question,
            "options": options,
            "answer": correct[0],
            "meta": {
                "impression_id": impression.entity_id,
                "impression_category": impression.category,
                "impression_subcategory": impression.subcategory,
                "impression_concept": impression.concept,
                "finding_ids": [f.entity_id for f in selected_findings],
                "findings_used": [f.concept for f in selected_findings],
                "scope_level": "relationship (inference)",
            },
        }

    @staticmethod
    def _norm_val(v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @staticmethod
    def _difficulty_for_role(role: str) -> str:
        return {"perceptual": "D0", "topological": "D1", "epistemic": "D1"}.get(role, "D0")


# I/O functions
# Load reports from JSON or JSONL file
def load_reports(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2)  # Peek at start to determine format (if JSON array or JSONL)
        f.seek(0)  # Reset file pointer to start
        if head.strip().startswith("["):  # JSON array
            return json.load(f)
        else:  # JSONL
            return [json.loads(line) for line in f if line.strip()]

# Save QA items to JSONL file
def save_qa(items: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# Save QA items to pretty-printed JSON file
def save_qa_companion_pretty_json(items, path_json: str):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Diagnose/Validate QA (v4 - streamlined with subcategory-based rollups).")
    parser.add_argument("--in", dest="in_path", required=True, help="Input reports file.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output QA file.")
    parser.add_argument("--schema", dest="schema_path", required=False, help="Schema JSON file.")
    parser.add_argument("--seed", type=int, default=SCHEMA_CONFIG["seed"], help="Random seed.")
    args = parser.parse_args()

    reports = load_reports(args.in_path)

    cfg = dict(SCHEMA_CONFIG)
    cfg["seed"] = args.seed
    gen = DiagnoseValidateQAGenerator(cfg, schema_path=args.schema_path)
    qa = gen.generate_for_reports(reports)

    save_qa(qa, args.out_path)

    save_qa_companion_pretty_json(qa, args.out_path.replace(".jsonl", "_pretty.json"))

    print(f"Generated {len(qa)} QA items from {len(reports)} reports")
    
    by_type = collections.defaultdict(int)
    by_scope = collections.defaultdict(int)
    for item in qa:
        by_type[item["question_type"]] += 1
        scope_level = item["meta"].get("scope_level", "N/A")
        by_scope[scope_level] += 1
    
    print("\nQuestion types:")
    for qtype, count in sorted(by_type.items()):
        print(f"  {qtype}: {count}")
    
    print("\nScope levels:")
    for scope, count in sorted(by_scope.items()):
        print(f"  {scope}: {count}")