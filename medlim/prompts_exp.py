"""
Prompts for the 200-paper experiment (PROMPT_VERSION exp1.0):
  1. author-limitation EXTRACTION (atomic units, verbatim quote)          - open LLMs
  2. CONSENSUS of the three extractions                                    - gpt-4o-mini
  3. pseudo-label candidates: ADDITIONAL limitations with societal impact  - open LLMs
  4. generation with SOCIETAL IMPACT: zero-shot / RAG / sequential agents + leader
  5. judges: list matching, pairwise Yes/No (archived prompt), societal-impact rating
"""
from .prompts import block_paper, block_rag, block_ph, JSON_RULE, FALLBACK_RULE  # noqa: F401

PROMPT_VERSION = "exp1.1"

# ============================================================================ 1. extraction
EXTRACT_SYSTEM = (
    "You extract the limitations that the AUTHORS THEMSELVES acknowledge in a scientific paper. You are an "
    "extractor, not a reviewer: never add your own critique.")


def extract_user(title, text, k=15):
    return (f"PAPER TITLE: {title}\n\nPAPER TEXT (end matter removed; for long papers only the discussion part "
            f"and the passage around any limitations section are shown):\n{text}\n\n"
            "TASK: List every limitation, weakness or caveat of THIS study that the authors explicitly acknowledge.\n"
            "Rules:\n"
            "1. ATOMIC: one limitation per item. Split compound sentences ('First, ... Second, ...', 'small and "
            "single-centre sample') into separate items.\n"
            "2. Keep the authors' specifics (numbers, populations, design names); one sentence per item, in your "
            "words but faithful to the authors' meaning.\n"
            "3. Include limitations stated anywhere (limitations section, discussion, conclusion).\n"
            "4. EXCLUDE: limitations of previous studies or other methods, generic future-work wishes that do not "
            "state a weakness, the strengths of the study, and anything you infer yourself.\n"
            "5. For each item give a VERBATIM quote (<= 40 words) from the text that states it.\n"
            f"6. At most {k} items. If the authors acknowledge no limitation, return an empty list.\n"
            'JSON schema: {"limitations": [{"limitation": "one sentence", "quote": "verbatim text"}]}\n' + JSON_RULE)


# ============================================================================ 2. consensus (gpt-4o-mini)
CONSENSUS_SYSTEM = (
    "You merge the AUTHOR-STATED limitations that three independent extractors (E1, E2, E3) found in the same "
    "paper into clusters for MAJORITY voting. Be LENIENT: put items in the same cluster whenever they express the "
    "same or a similar limitation - paraphrases, the same issue described with different words or detail, one item "
    "more specific or more general than the other, or items that overlap substantially (e.g. 'single-centre study' "
    "and 'participants were recruited from one hospital'). Keep items in separate clusters only when they are "
    "clearly different problems. Each extractor may contribute several items to one cluster. Every item id must "
    "appear in exactly one cluster.")


def consensus_user(title, lists: dict, passage=""):
    blocks = []
    for eid, items in lists.items():
        blocks.append(f"EXTRACTOR {eid}:\n" + ("\n".join(f"[{eid}_{j + 1}] {t}" for j, t in enumerate(items)) or "(none)"))
    ctx = f"\nAUTHORS' LIMITATION PASSAGE (may be incomplete, for wording only):\n{passage[:3000]}\n" if passage else ""
    return (f"PAPER TITLE: {title}\n{ctx}\n" + "\n\n".join(blocks) +
            "\n\nTASK: Cluster the items ([E?_?] ids). For each cluster write one canonical sentence that states the "
            "limitation as the authors did (atomic, specific), and list its member ids.\n"
            'JSON schema: {"clusters": [{"canonical": "...", "members": ["E1_2", "E3_1"]}]}\nReturn JSON only.')


# ============================================================================ 3/4. societal-impact generation
SOC_DEF = (
    "A limitation WITH SOCIETAL IMPACT is a specific weakness of the study (design, cohort, measurement, analysis, "
    "generalizability, deployment) together with its consequence for people: which population could be "
    "misserved, harmed, excluded or misled if the findings are applied (e.g. underrepresented racial/ethnic "
    "groups, rural or low-income communities, uninsured patients, areas with provider shortages), and how "
    "(biased care, unequal access, wrong policy, wasted resources). Methodological limitations without an obvious "
    "societal angle are still valid; then say who would be affected by acting on a wrong conclusion.")

GEN_ITEM_SCHEMA = ('{"limitations": [{"limitation": "one or two sentences: the weakness of THIS study", '
                   '"societal_impact": "who is affected and how", "affected_population": "short", '
                   '"evidence": "short verbatim quote from the paper or a [PH-n] tag"}]}')

ZS_SYSTEM = ("You are an expert peer reviewer of biomedical and public-health research. You identify the specific "
             "limitations of a study and their societal impact. " + SOC_DEF +
             " Be specific to THIS paper; avoid boilerplate such as 'more research is needed'.")


def zs_user(title, text, k, passages=None, ph_text=""):
    extra = []
    if passages:
        extra.append("Use the retrieved literature where it reveals a limitation (contradicting results, known "
                     "pitfalls of the design, populations other studies covered).")
    if ph_text:
        extra.append("Use the public-health data to show who in the US population would be affected (burden, "
                     "vulnerability, insurance, provider shortages) and cite the [PH-n] tag.")
    return (block_paper(title, text) + block_rag(passages) + block_ph(ph_text) +
            f"\n\nTASK: List the {k} most important limitations of this paper with their societal impact (at most "
            f"{k}, most important first). " + " ".join(extra) + f"\nJSON schema: {GEN_ITEM_SCHEMA}\n" + JSON_RULE)


PSEUDO_SYSTEM = ("You are a senior methodologist and public-health reviewer. The authors of a paper listed their "
                 "own limitations. Find ADDITIONAL, well-supported limitations with societal impact that the "
                 "authors did NOT mention. " + SOC_DEF + " Report only issues a careful expert would agree are real "
                 "and that the paper text or the public-health data support.")

PSEUDO_QUESTIONS = """Consider before answering:
Q1. Threats to validity in design, cohort selection, measurement or analysis that the authors did not acknowledge?
Q2. Does the study population (age, sex, race/ethnicity, insurance, geography, rurality) match the burden of the condition in CDC PLACES and the vulnerable groups in CDC SVI / Census ACS?
Q3. Would the intervention or model work where HRSA shortage areas or critical-access / low-rated CMS hospitals are common?
Q4. Missing subgroup analyses for groups that carry high burden?
Q5. Claims (efficacy, generalizability, causality, policy recommendations) stronger than the evidence?"""


def pseudo_user(title, text, author_lims, k, ph_text=""):
    auth = "\n".join(f"- {a}" for a in author_lims) if author_lims else "(the authors did not state any limitations)"
    return (block_paper(title, text) + block_ph(ph_text) +
            f"\n\nLIMITATIONS ALREADY STATED BY THE AUTHORS (do NOT repeat or paraphrase them):\n{auth}\n\n"
            + PSEUDO_QUESTIONS +
            f"\n\nTASK: List up to {k} ADDITIONAL limitations with societal impact. Empty list if none.\n"
            f"JSON schema: {GEN_ITEM_SCHEMA}\n" + JSON_RULE)


# ---------------------------------------------------------------------------- sequential agents
AGENT_COMMON_SYSTEM = ("You are one specialist reviewer in a team that critiques biomedical papers for limitations "
                       "with societal impact. Your role is given at the END of the user message. Stay inside that "
                       "role and do not repeat findings that teammates already reported.")

# name, dimension, sees public-health data, role
AGENTS = [
    ("CohortAgent", "cohort", True,
     "CohortAgent: demographic narrowness, missing subgroups and strict exclusion criteria. With public-health data, "
     "say how the cohort differs from the affected US population (burden, vulnerability). Quote exclusion criteria."),
    ("FairnessAgent", "fairness", True,
     "FairnessAgent: missing subgroup analyses and disparate performance or effects by race/ethnicity, sex, age, "
     "income, insurance or rurality. Name the subgroup, the metric and the gap; use SVI/ACS data to weight harm."),
    ("StatRigorAgent", "stats_rigor", False,
     "StatRigorAgent: sample size and power, leakage (train/test, temporal, site), confidence intervals, multiple "
     "comparisons, confounding, weak baselines. Quote the passage that creates the risk."),
    ("EfficacyOverclaimAgent", "efficacy_overclaim", False,
     "EfficacyOverclaimAgent: claims of efficacy, causality or policy relevance stronger than the evidence, using "
     "retrieved literature when available. Cite the claim and the missing or contradicting evidence."),
    ("DeploymentAgent", "deployment", True,
     "DeploymentAgent: operational assumptions (infrastructure, specialists, EHR integration, cost, regulation, "
     "prospective validation). Use CMS hospital and HRSA shortage data: low-resource settings often lack them."),
    ("CitationFidelityAgent", "citation", False,
     "CitationFidelityAgent: claims attributed to prior work that the retrieved literature does not support, or "
     "retrieved papers that contradict this paper. Quote both sides; emit nothing when unsure."),
    ("MethodologyAgent", "methodology", False,
     "MethodologyAgent: study design, measurement tools, data collection, follow-up, protocol decisions that "
     "threaten validity (not sample size, generalizability or translation)."),
    ("ClinicalTranslationAgent", "clinical_translation", False,
     "ClinicalTranslationAgent: obstacles to applying the findings in routine care (animal/in-vitro models, narrow "
     "disease stages, artificial settings, adherence, cost)."),
    ("ContaminationControlAgent", "contamination_control", False,
     "ContaminationControlAgent: contamination, sample processing, data cleaning, missing-data handling, batch "
     "effects, quality control, insufficient controls."),
    ("GeneralizabilityPowerAgent", "generalizability_power", True,
     "GeneralizabilityPowerAgent: representativeness (geography, setting, demographics, selection bias) and power "
     "for subgroups. Use public-health data to show which populations the results may not cover."),
    ("EcologicalFallacyAgent", "ecological_fallacy", True,
     "EcologicalFallacyAgent: aggregation problems (ecological fallacy, group-to-individual inference, Simpson's "
     "paradox). Public-health data are area-level: do not commit the fallacy yourself."),
]

AGENT_JSON = ("\n\nOUTPUT: JSON only: {\"limitations\": [{\"claim\": \"one specific sentence\", \"societal_impact\": "
              "\"who is affected and how\", \"affected_population\": \"short\", \"evidence_span\": \"VERBATIM quote "
              "from the paper / a retrieved paper, or a [PH-n] line\", \"evidence_source\": \"manuscript | corpus | "
              "public_data\", \"ph_tags\": [\"PH-2\"], \"severity\": \"low | medium | high | critical\", "
              "\"confidence\": 0.0-1.0}]}. At most 3 items; an empty list if nothing new is in your scope.")


def agent_user(title, text, role, prior, passages=None, ph_text=""):
    """paper first (shared prefix for vLLM prefix caching), then teammates' findings, then the role."""
    done = ("\n".join(f"- ({p['agent']}) {p['claim']}" for p in prior) if prior else "(none yet)")
    return (block_paper(title, text) + block_rag(passages) + block_ph(ph_text) +
            f"\n\nFINDINGS ALREADY REPORTED BY TEAMMATES (do not repeat; you may add a different angle):\n{done}"
            f"\n\nYOUR ROLE: {role}" + AGENT_JSON)


LEADER_SYSTEM = ("You are the LEADER agent. You receive the critiques of eleven specialist reviewers of one paper and "
                 "write the FINAL set of limitations with societal impact. " + SOC_DEF)


def leader_user(title, abstract, candidates, k, ph_text=""):
    lines = "\n".join(f"[C{i + 1}] ({c.get('agent', '?')}, severity={c.get('severity', '?')}, "
                      f"anchored={c.get('anchored')}) {c.get('claim', '')} | impact: {c.get('societal_impact', '')}"
                      for i, c in enumerate(candidates))
    return (f"PAPER TITLE: {title}\nABSTRACT: {abstract[:2500]}" + block_ph(ph_text) +
            f"\n\nSPECIALIST CRITIQUES:\n{lines or '(none)'}\n\n"
            f"TASK: Return at most {k} final limitations, most important first. Merge duplicates, drop boilerplate "
            "and unsupported items (prefer anchored=True), keep each item atomic, and state its societal impact "
            + ("using the public-health figures where relevant ([PH-n])" if ph_text else "") +
            ". List the candidate ids each final item is based on.\n"
            'JSON schema: {"limitations": [{"limitation": "...", "societal_impact": "...", "affected_population": '
            '"...", "source_ids": ["C3", "C7"]}]}\nReturn JSON only.')


# ============================================================================ 5. judges
JUDGE_LIST_SYSTEM = ("You are a strict evaluator of scientific peer-review critiques. You decide which generated "
                     "limitations describe the same underlying limitation as a reference limitation.")


def judge_list_user(refs, gens):
    r = "\n".join(f"[R{i + 1}] {t}" for i, t in enumerate(refs))
    g = "\n".join(f"[G{j + 1}] {t}" for j, t in enumerate(gens))
    return (f"REFERENCE LIMITATIONS:\n{r}\n\nGENERATED LIMITATIONS:\n{g}\n\n"
            "For EVERY generated limitation, list the reference ids that express the same or a closely similar "
            "limitation (same underlying problem; wording may differ; one generated item may match several "
            "references; [] if none). Do not match on vague overlap such as both mentioning 'sample'.\n"
            'Respond with JSON only: {"matches": [{"g": "G1", "r": ["R2"]}, {"g": "G2", "r": []}]}')


IMPACT_SYSTEM = "You rate how well each limitation states a concrete societal impact."


def impact_user(items):
    body = "\n".join(f"[G{i + 1}] {it.get('limitation', '')} || impact: {it.get('societal_impact', '') or '(none)'} "
                     f"|| population: {it.get('affected_population', '') or '(none)'}" for i, it in enumerate(items))
    return (f"LIMITATIONS:\n{body}\n\nScore each item: 2 = names a specific affected population AND a plausible "
            "mechanism of harm/misuse; 1 = generic impact (e.g. 'limits generalizability') or population without "
            "mechanism; 0 = no societal impact. Also flag data_grounded=true when the impact cites a concrete "
            "figure or [PH-n] public-health line.\n"
            'Respond with JSON only: {"scores": [{"g": "G1", "score": 2, "data_grounded": false}]}')