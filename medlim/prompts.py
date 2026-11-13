"""All prompts in one place (versioned: change PROMPT_VERSION when you edit a prompt)."""

PROMPT_VERSION = "v2.0"

JSON_RULE = ("Return VALID JSON ONLY (no prose, no markdown fences). If you cannot produce JSON, write one "
             "line per item in the form 'Limitation 1: ...', 'Limitation 2: ...'.")

FALLBACK_RULE = ("Write ONLY the list, one item per line, exactly in this form:\n"
                 "Limitation 1: <one or two sentences>\nLimitation 2: <one or two sentences>\n...")

# ----------------------------------------------------------------------------- shared context blocks
def block_paper(title, text):
    return f"PAPER TITLE: {title}\n\nPAPER TEXT (author limitation section removed; may be truncated):\n{text}"


def block_rag(passages):
    if not passages:
        return ""
    body = "\n\n---\n\n".join(f"[similar paper {i + 1}]\n{p}" for i, p in enumerate(passages))
    return ("\n\nRETRIEVED LITERATURE (top similar papers from a PubMed corpus, hybrid FAISS+BM25; use as "
            "external comparators: precedents, contradictions, baselines, recurring limitations):\n" + body)


def block_ph(ph_text):
    if not ph_text:
        return ""
    return ("\n\nPUBLIC-HEALTH DATA (local copies of CDC PLACES, CDC SVI, Census ACS, CMS Hospital Compare, "
            "HRSA shortage areas). Each line has a tag like [PH-2]; cite the tag when a limitation relies on "
            "it. Use these figures to judge whether the study population, setting and intervention match the "
            "US population that would be affected (disease burden, social vulnerability, insurance, provider "
            "shortages, hospital capacity):\n" + ph_text)


# ----------------------------------------------------------------------------- zero-shot / RAG baselines
ZS_SYSTEM = ("You are an expert peer reviewer of biomedical and clinical research papers. You identify the "
             "specific limitations of a study: problems in design, cohort, measurement, statistics, "
             "generalizability, equity and real-world deployment. Be specific to THIS paper; avoid boilerplate "
             "such as 'more research is needed'.")


def zs_user(title, text, k, passages=None, ph_text=""):
    extra = []
    if passages:
        extra.append("Use the retrieved literature where it reveals a limitation of this paper.")
    if ph_text:
        extra.append("Use the public-health data where it reveals a gap between the study population/setting "
                     "and the affected US population; mention the [PH-n] tag you rely on.")
    return (block_paper(title, text) + block_rag(passages) + block_ph(ph_text) +
            f"\n\nTASK: List the {k} most important limitations of this paper (at most {k}). "
            + " ".join(extra) +
            '\nJSON schema: {"limitations": [{"limitation": "one or two sentences", '
            '"evidence": "short supporting quote or [PH-n] tag"}]}\n' + JSON_RULE)


# ----------------------------------------------------------------------------- pseudo-reference generator
PSEUDO_SYSTEM = ("You are a senior methodologist and public-health reviewer. The authors of a paper have "
                 "listed their own limitations. Your job is to find ADDITIONAL, well-supported limitations the "
                 "authors did NOT mention. Only report issues that a careful expert reviewer would agree are "
                 "real and that are supported by the paper text or by the provided public-health data.")

PSEUDO_QUESTIONS = """Answer these questions internally before writing the list:
Q1. Is anything in the design, cohort selection, measurement or analysis a threat to validity that the authors did not acknowledge?
Q2. Does the study population (age, sex, race/ethnicity, insurance, geography, rurality) match the burden of the target condition in CDC PLACES and the vulnerable groups identified by CDC SVI / Census ACS?
Q3. Would the intervention or model be feasible where HRSA shortage areas, critical-access or low-rated CMS hospitals are common?
Q4. Are subgroup analyses missing for groups that the public-health data show carry high burden?
Q5. Are claims (efficacy, generalizability, causality) stronger than the evidence supports?"""


def pseudo_user(title, text, author_lims, k, ph_text="", passages=None):
    auth = "\n".join(f"- {a}" for a in author_lims) if author_lims else "(the authors did not state any limitations)"
    return (block_paper(title, text) + block_rag(passages) + block_ph(ph_text) +
            f"\n\nLIMITATIONS ALREADY STATED BY THE AUTHORS (do NOT repeat or paraphrase these):\n{auth}\n\n"
            + PSEUDO_QUESTIONS +
            f"\n\nTASK: List up to {k} ADDITIONAL limitations not covered by the authors' list. Each must be "
            "specific to this paper. If there are none, return an empty list.\n"
            'JSON schema: {"limitations": [{"limitation": "one or two sentences", '
            '"basis": "manuscript | public_data | literature", "evidence": "short quote or [PH-n] tag"}]}\n'
            + JSON_RULE)


# ----------------------------------------------------------------------------- majority vote (gpt-4o-mini)
VOTE_SYSTEM = ("You CLUSTER limitation statements written by several independent reviewers for MAJORITY "
               "voting. Two statements belong to the same cluster only if they describe the same underlying "
               "problem of the paper (same threat, same population/aspect), even if worded differently. Do "
               "not merge merely related topics. Every statement id must appear in exactly one cluster.")


def vote_user(title, author_lims, lists: dict):
    auth = "\n".join(f"[A{i + 1}] {a}" for i, a in enumerate(author_lims)) or "(none)"
    blocks = []
    for mid, items in lists.items():
        blocks.append(f"REVIEWER {mid}:\n" + ("\n".join(f"[{mid}_{j + 1}] {t}" for j, t in enumerate(items)) or "(none)"))
    return (f"PAPER TITLE: {title}\n\nAUTHOR-STATED LIMITATIONS:\n{auth}\n\n" + "\n\n".join(blocks) +
            "\n\nTASK: Group the reviewer statements ([M?_?] ids) into clusters of the same limitation. For each "
            "cluster give one canonical sentence that states the limitation precisely, the member ids, and "
            "duplicates_author=true if the cluster restates any author-stated limitation [A?].\n"
            'JSON schema: {"clusters": [{"canonical": "...", "members": ["M1_2", "M3_1"], '
            '"duplicates_author": false, "author_ids": []}]}\nReturn JSON only.')


# ----------------------------------------------------------------------------- 11 specialist agents
AGENT_JSON = ("\n\nOUTPUT: JSON only, schema {\"limitations\": [{\"dimension\": \"<your dimension>\", \"claim\": "
              "\"one specific sentence\", \"evidence_span\": \"VERBATIM quote from the paper text, a retrieved "
              "paper, or a [PH-n] line\", \"evidence_source\": \"manuscript | corpus | public_data\", "
              "\"ph_tags\": [\"PH-2\"], \"affected_subgroup\": \"who is harmed or 'any'\", \"severity\": "
              "\"low | medium | high | critical\", \"confidence\": 0.0-1.0}]}. Emit at most 3 items; emit an empty "
              "list if you find nothing in your scope. Never paraphrase inside evidence_span.")

AGENTS = [
    # name, dimension, uses public-health data, system prompt  (6 PolicyGuard + 5 domain, as in v1)
    ("CohortAgent", "cohort", True,
     "You are CohortAgent. You find demographic narrowness, missing subgroups and strict exclusion criteria in "
     "the study cohort. When public-health data are given, quantify how the cohort differs from the affected US "
     "population (burden, vulnerability), not only percentages. Ignore generic concerns; quote exclusion "
     "criteria verbatim. If the cohort matches the target population, emit nothing."),
    ("FairnessAgent", "fairness", True,
     "You are FairnessAgent. You find missing subgroup performance analysis and disparate performance signals. "
     "Use SVI / ACS data to weight harm: a gap in a high-vulnerability population is more severe. Always name "
     "the subgroup, the metric and the gap. Never write 'fairness concerns' without saying what, who and how much."),
    ("StatRigorAgent", "stats_rigor", False,
     "You are StatRigorAgent. You check sample-size adequacy, train/test/temporal/site leakage, confidence "
     "intervals, multiple-comparison correction and weak baselines. Quote the methodological passage that "
     "creates each risk. Only emit if a statistical decision could mislead a reader or a deployment committee."),
    ("EfficacyOverclaimAgent", "efficacy_overclaim", False,
     "You are EfficacyOverclaimAgent. You compare the paper's claimed efficacy with prior results (retrieved "
     "literature when available) and with its own evidence. Emit only when a claim materially exceeds what the "
     "evidence supports. Cite the claim and the contradicting or missing evidence."),
    ("DeploymentAgent", "deployment", True,
     "You are DeploymentAgent. You audit operational assumptions: compute, infrastructure, EHR integration, "
     "specialist availability, regulatory pathway and prospective validation. Cross-reference CMS hospital and "
     "HRSA shortage data: critical-access and shortage-area facilities often lack the assumed resources."),
    ("CitationFidelityAgent", "citation", False,
     "You are CitationFidelityAgent. You check whether claims attributed to prior work are consistent with the "
     "retrieved literature, and whether any retrieved paper contradicts a claim of this paper. Quote both "
     "sides. False positives must be near zero; emit nothing when unsure."),
    ("MethodologyAgent", "methodology", False,
     "You are MethodologyAgent. Identify limitations in study design, measurement tools, data collection, "
     "participant engagement, experimental procedures or protocol decisions. Quote the passage and explain how "
     "it threatens validity. Do not discuss sample size, generalizability, contamination or clinical translation."),
    ("ClinicalTranslationAgent", "clinical_translation", False,
     "You are ClinicalTranslationAgent. Identify limitations that constrain applicability to real-world clinical "
     "settings: animal or in-vitro models, narrow disease stages, artificial conditions, assumptions that do not "
     "hold in routine care. Quote the passage and explain why findings may not translate."),
    ("ContaminationControlAgent", "contamination_control", False,
     "You are ContaminationControlAgent. Identify limitations from contamination, sample processing, data "
     "cleaning, inclusion/exclusion handling, batch effects, quality control or insufficient controls. Quote the "
     "methods/results passage and explain the reliability impact."),
    ("GeneralizabilityPowerAgent", "generalizability_power", True,
     "You are GeneralizabilityPowerAgent. Identify (a) generalizability problems: representativeness, demographic "
     "or geographic restriction, selection bias, narrow inclusion; and (b) statistical power problems: small "
     "samples, no subgroup analysis, lack of diversity. Use public-health data to show which populations the "
     "results may not cover."),
    ("EcologicalFallacyAgent", "ecological_fallacy", True,
     "You are EcologicalFallacyAgent. Identify inferential problems from aggregation: ecological fallacy, "
     "group-to-individual (or reverse) extrapolation, causal claims from aggregated data, Simpson's paradox risk. "
     "Public-health data are area-level; do not commit the fallacy yourself."),
]


AGENT_COMMON_SYSTEM = ("You are one specialist reviewer in a team that critiques biomedical papers. Your role "
                       "is given at the END of the user message. Stay strictly inside that role.")


def agent_user(title, text, role_prompt, passages=None, ph_text=""):
    """Paper first, role last: the 11 agents share the same token prefix, so vLLM prefix caching
    computes the paper only once per paper."""
    return (block_paper(title, text) + block_rag(passages) + block_ph(ph_text) +
            "\n\nYOUR ROLE: " + role_prompt +
            "\nEmit the limitations in YOUR scope only. evidence_source='manuscript' needs a verbatim quote of "
            "the paper text; 'corpus' a verbatim quote of a retrieved paper; 'public_data' the [PH-n] tag(s)."
            + AGENT_JSON)


MERGE_SYSTEM = ("You are the Leader agent that MERGEs critiques from specialist reviewers of one paper. "
                "Deduplicate (same underlying concern = one item), drop weak or boilerplate items, and keep the "
                "most important ones. Do not invent new content.")


def merge_user(title, candidates, k):
    lines = "\n".join(f"[C{i + 1}] ({c.get('dimension', '?')}, severity={c.get('severity', '?')}) {c.get('claim', '')}"
                      for i, c in enumerate(candidates))
    return (f"PAPER TITLE: {title}\n\nCANDIDATE LIMITATIONS:\n{lines}\n\nTASK: Return at most {k} final "
            "limitations, most important first. Each has one clear claim (you may merge wording of duplicates) "
            "and the ids of the candidates it is based on.\n"
            'JSON schema: {"limitations": [{"claim": "...", "source_ids": ["C3", "C7"]}]}\nReturn JSON only.')


# ----------------------------------------------------------------------------- judges (gpt-4o-mini)
def judge_pair_prompt(gt_text, llm_text):
    """The archived pairwise Yes/No prompt (verbatim)."""
    description1 = f"ground truth limitations: {gt_text}"
    description2 = f"llm generated limitations: {llm_text}"
    return ("Check whether 'list2' contains a topic or limitation from 'list1' "
            "or 'list1' contains a topic or limitation from 'list2'.\n\n"
            "Your answer should be \"Yes\" or \"No\".\n"
            f"List 1: {description1}\n"
            f"List 2: {description2}\n")


JUDGE_LIST_SYSTEM = ("You are a strict evaluator of scientific peer-review critiques. You decide which generated "
                     "limitations describe the same underlying limitation as a reference limitation.")


def judge_list_user(refs, gens):
    r = "\n".join(f"[R{i + 1}] {t}" for i, t in enumerate(refs))
    g = "\n".join(f"[G{j + 1}] {t}" for j, t in enumerate(gens))
    return (f"REFERENCE LIMITATIONS:\n{r}\n\nGENERATED LIMITATIONS:\n{g}\n\n"
            "For EVERY generated limitation, list the reference ids that express the same limitation or topic "
            "(same problem; wording may differ; a generated item may match several references; use [] if none). "
            "Do not match on vague overlap such as both mentioning 'sample'.\n"
            'Respond with JSON only: {"matches": [{"g": "G1", "r": ["R2"]}, {"g": "G2", "r": []}]}')
