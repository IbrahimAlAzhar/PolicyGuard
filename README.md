
Limitations with societal impact, scored against **author-stated limitations only**. There are no pseudo labels.

## Pipeline
| step | PBS | GPU | what happens |
|---|---|---|---|
| 1a | `pbs/1_extract_mistral7b.pbs` | Mistral-7B-Instruct-v0.3, bf16 | author limitations (atomic units + verbatim quote) for the 500 sampled papers |
| 1b | `pbs/1_extract_qwen32b.pbs` | Qwen3-32B, 4-bit | the same, second extractor |
| 2 | `pbs/2_build_gt_gpt.pbs` | CPU | gpt-4o-mini merges both lists **leniently** into Limitation Units; GT = units found by both (`GT_RULE=both`) or either (`union`); the 300 papers with the most units are selected |
| 3 | `pbs/3_gen_eval_{qwen8b,gemma27b,mistral24b}.pbs` | 1 model each | 6 conditions on the 300 papers, then the gpt-4o-mini holistic judge + NLP metrics |

Conditions: zero-shot / RAG / LLM agents, each **without** and **with** CDC/HRSA data. RAG and agents both use 3
retrieved similar papers. Agents = 5 specialists run in sequence (Methods & Statistics, Population &
Generalizability, Equity & Societal Impact, Translation & Deployment, Evidence & Claims) plus a leader. Every
condition returns at most 8 limitations.

Evaluation: one gpt-4o-mini call per paper × model × condition.
* Recall = ground-truth units covered by at least one generated item.
* Precision = generated items that match at least one ground-truth unit.
* F1 is computed from these two.

The NLP metrics are ROUGE-1, ROUGE-L, TF-IDF cosine, embedding soft-F1 (all-MiniLM-L6-v2) and, optionally,
BERTScore (set `MEDLIM_BERTSCORE_MODEL` to a local roberta-large folder).

## Run
```bash
cd med_limagents_v2
mkdir -p outputs && sed -i 's/\r$//' pbs/*.pbs pbs/*.sh *.py medlim/*.py
source activate image_lim_vllm_final
python medlim_v3.py --stage dry                                  # no LLM: sample + outputs/main/dry_check.txt
RUN_NAME=test N_SAMPLE=10 N_SELECT=5 bash pbs/submit_all.sh       # small end-to-end test
bash pbs/submit_all.sh                                           # full: 500 sampled -> 300 selected
```
`submit_all.sh` chains the jobs: 1a and 1b run in parallel, then step 2, then the three step-3 jobs in parallel.

## Output: `outputs/<RUN_NAME>/`
* `medlim_v3_results.csv`: the **one** results table. One row per sampled paper, containing the paper, retrieval,
  CDC/HRSA context, `ext_mistral7b`, `ext_qwen32b`, `gt_units` (gold standard), `n_gt`, `selected`, then for
  every model and condition `<model>__<cond>` (the generated limitations) and `__precision __recall __f1 __rouge1 ...`.
* `final_text.txt`: all tables (ground truth, main results, 2×3 ablation grid, paired effects, diagnostics).
* `work/`: each job's own part file (merged into the results table automatically). There's no need to open it.
* `logs/`: run logs and vLLM logs.

