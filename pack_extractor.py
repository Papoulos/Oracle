import os
import json
import logging
import datetime
from collections import Counter
from collections.abc import Iterable
from pydantic import ValidationError, TypeAdapter

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import difflib
import math
import re
from rapidfuzz import fuzz

import config
from base_utils import BaseAgent, extract_json
from systems.pdf_pages import PageText, read_pdf_pages, format_pages
from systems.promote import get_draft_dir
from systems.provenance import (
    ProvenanceData, ProvenanceEntry, Candidate, ContextPage,
    REVIEW_CONFIDENCE_THRESHOLD
)
from mechanics.models import (
    Manifest, ResolutionConfig, ResourcesConfig, TriggersConfig, FAMILIES,
    PbtA2d6Config
)
from systems.validate import validate_pack


def parse_confidence(raw) -> int | None:
    if isinstance(raw, bool):
        return None

    if isinstance(raw, (int, float)):
        if math.isnan(raw) or math.isinf(raw):
            return None
        val = int(round(raw))
        return max(0, min(100, val))

    if isinstance(raw, str):
        # Allow optional spaces and an optional % at the end.
        m = re.fullmatch(r"^\s*(\d{1,3})(?:\.\d+)?\s*%?\s*$", raw)
        if m:
            val = round(float(raw.strip().replace('%', '')))
            return max(0, min(100, val))

    return None

def normalize_language(raw) -> str | None:
    if not isinstance(raw, str):
        return None

    s = raw.strip().lower()
    m = re.fullmatch(r"([a-z]{2})(?:[-_][a-z0-9]+)*", s)
    if m:
        return m.group(1)
    return None


def select_pages(store: Chroma, queries: list[str], pages: list[PageText], max_chars: int, k: int, neighbors: int = 1) -> list[PageText]:
    """
    Selects relevant pages using RAG similarity search and returns a continuous,
    deduplicated block of PageText up to max_chars in length.
    """
    if store is None:
        raise ValueError("Chroma store is None but full text exceeds MAX_CHARS.")

    # 1. Gather relevant page markers from Chroma
    hit_counts = Counter()
    for q in queries:
        docs = store.similarity_search(q, k=k)
        for d in docs:
            # We skip documents missing page metadata (e.g., indexed json)
            if "page" not in d.metadata or "source" not in d.metadata:
                continue

            # Chroma/PyPDFLoader pages are 0-indexed, ours are 1-indexed
            basename = os.path.basename(d.metadata["source"])
            page_num = int(d.metadata["page"]) + 1
            hit_counts[(basename, page_num)] += 1

    # Sort hits by frequency
    sorted_hits = sorted(hit_counts.items(), key=lambda x: x[1], reverse=True)

    selected_keys = set()
    total_chars = 0

    # Fast lookup for our pages
    pages_dict = {(p.source, p.page): p for p in pages}

    for (src, page_num), _ in sorted_hits:
        # We try to add this page and its neighbors
        candidates = [(src, p) for p in range(page_num - neighbors, page_num + neighbors + 1)]
        for cand in candidates:
            if cand not in selected_keys and cand in pages_dict:
                p_text = pages_dict[cand]
                # length includes formatting
                formatted_len = len(f"[[{p_text.source} p.{p_text.page}]]\n{p_text.text}\n\n")
                if total_chars + formatted_len > max_chars and total_chars > 0:
                    # Don't add if it exceeds max_chars (unless it's the very first page)
                    break
                selected_keys.add(cand)
                total_chars += formatted_len
        if total_chars > max_chars:
             break

    # Reconstruct final list of selected PageText, sorted naturally
    selected_pages = [pages_dict[k] for k in selected_keys]
    selected_pages.sort(key=lambda p: (p.source, p.page))
    return selected_pages


STEP_A_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert RPG systems analyst.
Determine the mechanical resolution family of the provided rulebook, its name, and its language.
The possible families are: {families}

If you are absolutely certain, return:
```json
{{
  "family": "NameOfFamily",
  "confidence": 95,
  "why": "...",
  "name": "Name of the RPG System",
  "language": "en"
}}
```
`language` MUST be a valid ISO 639-1 two-letter code (e.g., "fr", "en").

If you are unsure about the family, you MUST NOT guess. Return family "unknown" and list candidates:
```json
{{
  "family": "unknown",
  "reason": "Explain why it's ambiguous...",
  "candidates": [
    {{"family": "CandidateFamily1", "confidence": 50, "why": "..."}}
  ],
  "name": "Name of the RPG System",
  "language": "en"
}}
```
Note: Any invalid or unparseable `confidence` value will be counted as 0.
"""),
    ("human", "CODEX EXCERPTS:\n{context}"),
])

STEP_B_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert RPG systems configurator.
Generate `resolution.json`, `resources.json`, and `triggers.json` for the rules.

Target language for the extracted strings: {language}

Below is the expected JSON Schema for {family} in resolution.json:
{resolution_schema}

Below is the expected JSON Schema for resources.json:
{resources_schema}

Below is the expected JSON Schema for triggers.json:
{triggers_schema}

IMPORTANT:
For each file you generate, you MUST also provide provenance entries.
Your output MUST be a JSON object with this exact structure:
```json
{{
  "resolution.json": {{
    "config": {{ ... valid resolution config ... }},
    "provenance": [
      {{"path": "/advantage_enabled", "source": "rules.pdf", "page": 12, "excerpt": "exact phrase from text (<=25 words)", "confidence": 90}}
    ]
  }},
  "resources.json": {{ ... }},
  "triggers.json": {{ ... }}
}}
```

The "path" MUST be a valid RFC 6901 JSON Pointer to the leaf scalar value in the config object (e.g. "/tiers/0/outcome" or "/multiplier"). Do not provide provenance for structural arrays/objects, only the actual scalar values.
The "excerpt" MUST be a direct quote from the text that justifies the value, and MUST NOT exceed 25 words.
"""),
    ("human", "CODEX EXCERPTS:\n{context}\n\nPREVIOUS ERRORS (if any):\n{errors}"),
])


class PackExtractorAgent(BaseAgent):
    def __init__(self, store: Chroma | None = None, verbose=False):
        # We use a strict temperature for data extraction tasks
        super().__init__(model=config.ORCHESTRATOR_MODEL, temperature=0.1, verbose=verbose)
        self.store = store
        self.queries_fr = [
            "résolution d'action, jet de dé, réussite, échec, critique",
            "points de vie, ressources, magie, mana, blessures",
            "repos court, repos long, récupération, soins, downtime"
        ]
        self.queries_en = [
            "action resolution, dice roll, success, failure, critical",
            "hit points, health, resources, magic, mana, wounds",
            "short rest, long rest, recovery, healing, downtime"
        ]

    def _normalize_text(self, text: str) -> str:
        import unicodedata
        import re
        t = unicodedata.normalize('NFKD', text)
        t = t.lower()
        t = re.sub(r'-\n', '', t)
        t = re.sub(r'[\s]+', ' ', t)
        return t.strip()

    def _fuzzy_match(self, excerpt: str, page_text: str) -> bool:
        norm_exc = self._normalize_text(excerpt)
        norm_page = self._normalize_text(page_text)
        if not norm_exc: return False

        if norm_exc in norm_page:
            return True

        if len(norm_exc) < 10:
             return False

        ratio = fuzz.partial_ratio(norm_exc, norm_page)
        return ratio >= 88

    def _get_leaf_paths(self, data, current_path=""):
        paths = set()
        if isinstance(data, dict):
            for k, v in data.items():
                paths.update(self._get_leaf_paths(v, f"{current_path}/{k}"))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                paths.update(self._get_leaf_paths(v, f"{current_path}/{i}"))
        elif data is not None:
            paths.add(current_path)
        return paths

    def _evaluate_provenance(self, config_data: dict, raw_provenance: list, pages: list[PageText]) -> list[ProvenanceEntry]:
        leaf_paths = self._get_leaf_paths(config_data)

        prov_map = {p.get("path"): p for p in raw_provenance if isinstance(p, dict) and "path" in p}

        pages_dict = {(p.source, p.page): p.text for p in pages}

        final_entries = []

        for path in leaf_paths:
            if not path: continue # Skip empty root
            if path not in prov_map:
                final_entries.append(ProvenanceEntry(
                    file="", # Set by caller
                    path=path,
                    source="",
                    page=0,
                    excerpt="",
                    confidence=0,
                    needs_review=True,
                    review_reasons=["defaulted_by_schema"]
                ))
                continue

            p_data = prov_map[path]
            entry = ProvenanceEntry(
                file="", # Set by caller
                path=path,
                source=p_data.get("source", ""),
                page=p_data.get("page", 0),
                excerpt=p_data.get("excerpt", ""),
                confidence=p_data.get("confidence", 0)
            )

            reasons = []
            if entry.confidence < REVIEW_CONFIDENCE_THRESHOLD:
                reasons.append(f"low_confidence_under_{REVIEW_CONFIDENCE_THRESHOLD}")

            page_text = pages_dict.get((entry.source, entry.page))
            if page_text is None:
                reasons.append("page_not_found")
            else:
                if not self._fuzzy_match(entry.excerpt, page_text):
                    reasons.append("excerpt_not_found_in_page")

            if reasons:
                entry.needs_review = True
                entry.review_reasons = reasons

            final_entries.append(entry)

        return final_entries


    def extract(self, pack_id: str, files: list[str], override_language: str = None) -> None:
        draft_dir = get_draft_dir(pack_id)
        os.makedirs(draft_dir, exist_ok=True)

        logging.info(f"Reading PDFs for pack {pack_id}...")
        all_pages = read_pdf_pages(files)
        if not all_pages:
            raise ValueError(f"No text extracted from the provided files: {files}")

        full_text_len = sum(len(p.text) for p in all_pages)

        selected_pages = all_pages
        if full_text_len > config.PACK_CONTEXT_MAX_CHARS:
            if self.store is None:
                 raise ValueError("Text exceeds PACK_CONTEXT_MAX_CHARS and Chroma store is not initialized. Run `python indexer.py --core` first.")
            logging.info(f"Text too long ({full_text_len} chars). Selecting pages via RAG...")
            queries = self.queries_fr + self.queries_en
            selected_pages = select_pages(self.store, queries, all_pages, config.PACK_CONTEXT_MAX_CHARS, k=20, neighbors=1)

        if not selected_pages:
            raise ValueError("Core index is empty or does not contain these PDFs. Run `python indexer.py --core` first.")

        context_str = format_pages(selected_pages)
        context_pages_meta = [ContextPage(source=p.source, page=p.page) for p in selected_pages]

        # Step A: Family
        families_list = ", ".join(FAMILIES.keys())
        resp_a = self._invoke_logged(STEP_A_PROMPT, {"families": families_list, "context": context_str}, label="step_A_family")

        result_a = extract_json(resp_a.content, expected_type=dict)
        if not result_a:
             raise ValueError("Failed to parse JSON for Step A")

        family = result_a.get("family")
        sys_name = result_a.get("name", pack_id)

        # Language normalization
        language_raw = result_a.get("language")
        language_defaulted = False

        language = normalize_language(language_raw)
        if language is None:
            language = "en"
            language_defaulted = True

        if override_language:
            language = override_language
            language_defaulted = False

        # We always initialize provenance
        prov_data = ProvenanceData(
            status="incomplete",
            pack_id=pack_id,
            model=config.ORCHESTRATOR_MODEL if config.ORCHESTRATOR_MODEL else "unknown_model",
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            context_pages=context_pages_meta
        )

        if language_defaulted:
            prov_data.entries.append(ProvenanceEntry(
                file="manifest.yaml",
                path="/language",
                source="",
                page=0,
                excerpt="",
                confidence=0,
                needs_review=True,
                review_reasons=["language_defaulted"]
            ))

        # Confidence parsing
        raw_confidence = result_a.get("confidence")
        confidence = parse_confidence(raw_confidence)

        invalid_confidence = False
        if confidence is None:
            invalid_confidence = True
            confidence = 0

        if family == "unknown" or family not in FAMILIES or confidence < REVIEW_CONFIDENCE_THRESHOLD:
            prov_data.status = "unknown_family"
            reasons = []

            if family not in FAMILIES and family != "unknown":
                reasons.append(f"Proposed family '{family}' is not in the supported FAMILIES list.")
            elif family == "unknown":
                reasons.append(result_a.get("reason", "Family not determined"))
            else:
                if invalid_confidence:
                    reasons.append(f"invalid_confidence: {repr(raw_confidence)} traitée comme 0")
                if confidence < REVIEW_CONFIDENCE_THRESHOLD:
                    reasons.append(f"Confidence {confidence} is below the threshold of {REVIEW_CONFIDENCE_THRESHOLD}.")

            prov_data.reason = " | ".join(reasons)

            candidates_raw = [c for c in (result_a.get("candidates") or []) if isinstance(c, dict)]

            if family != "unknown" and family in FAMILIES and confidence < REVIEW_CONFIDENCE_THRESHOLD:
                 candidates_raw.append({"family": family, "confidence": confidence, "why": result_a.get("why", "")})

            for c in candidates_raw:
                try:
                     cand_conf = parse_confidence(c.get("confidence"))
                     if cand_conf is None:
                         cand_conf = 0
                     c["confidence"] = cand_conf
                     prov_data.candidates.append(Candidate(**c))
                except Exception:
                     pass

            with open(os.path.join(draft_dir, "provenance.json"), "w", encoding="utf-8") as f:
                f.write(prov_data.model_dump_json(indent=2))
            logging.info("Family unknown. Written provenance.json and stopped.")
            return

        # Step B: Extraction Loop
        # We prepare schemas
        import json as _json
        ResConfigClass = FAMILIES[family]
        res_schema = _json.dumps(ResConfigClass.model_json_schema(), indent=2)
        rec_schema = _json.dumps(ResourcesConfig.model_json_schema(), indent=2)
        trig_schema = _json.dumps(TriggersConfig.model_json_schema(), indent=2)

        errors = ""
        max_retries = 3

        # We'll save the best valid configurations we have across retries
        valid_configs = {}
        all_entries = []

        for attempt in range(max_retries):
            logging.info(f"Step B: Extraction attempt {attempt + 1}/{max_retries}...")
            resp_b = self._invoke_logged(STEP_B_PROMPT, {
                "language": language,
                "family": family,
                "resolution_schema": res_schema,
                "resources_schema": rec_schema,
                "triggers_schema": trig_schema,
                "context": context_str,
                "errors": errors
            }, label=f"step_B_extract_{attempt}")

            result_b = extract_json(resp_b.content, expected_type=dict)

            if not result_b:
                errors = "Failed to extract valid JSON from your previous response."
                continue

            current_errors = []

            # File validation
            validators = {
                "resolution.json": TypeAdapter(ResolutionConfig),
                "resources.json": TypeAdapter(ResourcesConfig),
                "triggers.json": TypeAdapter(TriggersConfig)
            }

            for file_name, adapter in validators.items():
                file_data = result_b.get(file_name)
                if not file_data:
                    current_errors.append(f"Missing '{file_name}' object.")
                    continue

                config_dict = file_data.get("config")
                if not config_dict:
                    current_errors.append(f"Missing 'config' inside '{file_name}'.")
                    continue

                try:
                    # Enforce family inside resolution config to ensure polymorphism works
                    if file_name == "resolution.json":
                        config_dict["family"] = family
                    valid_obj = adapter.validate_python(config_dict)

                    # Update our best knowledge
                    valid_configs[file_name] = valid_obj

                    # Process provenance
                    raw_prov = file_data.get("provenance", [])
                    entries = self._evaluate_provenance(valid_obj.model_dump(mode="json"), raw_prov, selected_pages)
                    for e in entries:
                         e.file = file_name

                    # Remove old entries for this file, add new ones
                    all_entries = [e for e in all_entries if e.file != file_name]
                    all_entries.extend(entries)

                except ValidationError as ve:
                    # Condense error
                    for err in ve.errors():
                        loc = ".".join(str(l) for l in err["loc"])
                        msg = err["msg"]
                        val = str(err.get("input", ""))[:80]
                        current_errors.append(f"File {file_name}, Path {loc}: {msg}. Value provided: {val}")

            if not current_errors:
                # We have all files valid individually. Now cross-file validation.
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    for fname, obj in valid_configs.items():
                         with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as f:
                              f.write(obj.model_dump_json())
                    # Write a mock manifest to satisfy validate_pack
                    man = Manifest.template(pack_id, family)
                    man.language = language
                    man.source_pdfs = [os.path.basename(f) for f in files]
                    with open(os.path.join(tmpdir, "manifest.yaml"), "w", encoding="utf-8") as f:
                         import yaml
                         yaml.dump(man.model_dump(), f)

                    issues = validate_pack(tmpdir)
                    error_issues = [i for i in issues if i.severity == "error"]

                    if error_issues:
                        for issue in error_issues:
                            current_errors.append(f"Cross-validation error in {issue.file} at {issue.path}: {issue.message}")
                    else:
                        # ALL GOOD! We have a complete, valid draft
                        prov_data.status = "complete"
                        break

            errors = "\n".join(current_errors)

        # End of retries or success
        if prov_data.status != "complete":
             prov_data.reason = errors if errors else "Max retries reached with unknown errors"

        # Keep initial entries (like language fallback) and add new ones
        prov_data.entries.extend(all_entries)

        # Write valid files
        for fname, obj in valid_configs.items():
             with open(os.path.join(draft_dir, fname), "w", encoding="utf-8") as f:
                  # Force ensure_ascii=False for accented characters
                  import json
                  json.dump(obj.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        # Write manifest
        if valid_configs: # Only if we got at least something
             man = Manifest.template(pack_id, family)
             man.name = sys_name
             man.version = "0.1.0-draft"
             man.language = language
             man.source_pdfs = [os.path.basename(f) for f in files]
             with open(os.path.join(draft_dir, "manifest.yaml"), "w", encoding="utf-8") as f:
                  import yaml
                  yaml.dump(man.model_dump(), f, allow_unicode=True)

        with open(os.path.join(draft_dir, "provenance.json"), "w", encoding="utf-8") as f:
             f.write(prov_data.model_dump_json(indent=2))

        if prov_data.status == "complete":
             logging.info(f"Successfully generated valid pack in {draft_dir}")
        else:
             logging.warning(f"Incomplete pack generated in {draft_dir}. Review provenance.json for errors.")
