"""Shared prompts and label configuration for NER and highlights tasks."""


LABELS = (
    "PROBLEM",
    "OBJECTIVE",
    "METHOD",
    "CONTRIBUTION",
    "FINDING",
    "IMPROVEMENT",
    "DATASET",
    "METRIC",
    "KEY_CONCEPT",
)

NER_SYSTEM = (
    "You are an expert scientific-paper annotator. You extract research-role "
    "entities from paper abstracts and output ONLY JSON."
)

NER_USER_TMPL = """Identify spans in the ABSTRACT that play a research role and label each with exactly one of these nine labels. Use the rarer labels (METRIC, DATASET, IMPROVEMENT) whenever they apply -- do not collapse everything into the common ones.

LABELS:
- PROBLEM - the gap, limitation, or challenge motivating the work.
- OBJECTIVE - the stated aim or goal.
- METHOD - an established/applied technique or procedure (NOT the novel contribution).
- CONTRIBUTION - the novel artifact the paper introduces (model, algorithm, system, compound, or newly released dataset).
- FINDING - a result or observation, stated WITHOUT an explicit comparison.
- IMPROVEMENT - a result stated as a comparative gain over a baseline/prior work ("2x faster", "higher than", "outperforms").
- DATASET - an existing data resource used or referenced (dataset, benchmark, corpus, cohort).
- METRIC - a named measured quantity or evaluation measure ("F1 score", "accuracy", "cell viability", "detection limit", "recovery rate").
- KEY_CONCEPT - a salient domain object/term that fits none of the above (a material, cell line, biological process, model component).

PRECEDENCE:
- A newly proposed thing is CONTRIBUTION even if it is also a method or dataset.
- A comparative result is IMPROVEMENT, not FINDING.
- A named measured quantity is METRIC, not KEY_CONCEPT.

RULES:
- Copy each surface string EXACTLY as it appears in the abstract (no paraphrase, no re-casing).
- List entities in order of appearance. Do not include offsets.

Return ONLY this JSON, no prose, no code fences:
{{"entities": [{{"text": "<exact surface>", "label": "<ONE LABEL>"}}]}}

EXAMPLE 1
ABSTRACT: Radiotherapy is a major modality for cancer treatment . Despite its therapeutic effect the acquired resistance of cancer cells to RT remains a significant clinical problem . Here we synthesized hydroxychloroquine loaded hollow mesoporous silica nanoparticles to enhance the therapeutic efficacy of RT . HCQ HMSN treated HCT116 cells showed a 200 fold higher intracellular uptake of HCQ than free HCQ .
OUTPUT: {{"entities": [{{"text": "Radiotherapy", "label": "METHOD"}}, {{"text": "acquired resistance of cancer cells to RT", "label": "PROBLEM"}}, {{"text": "hydroxychloroquine loaded hollow mesoporous silica nanoparticles", "label": "CONTRIBUTION"}}, {{"text": "enhance the therapeutic efficacy of RT", "label": "OBJECTIVE"}}, {{"text": "HCT116 cells", "label": "KEY_CONCEPT"}}, {{"text": "intracellular uptake", "label": "METRIC"}}, {{"text": "200 fold higher intracellular uptake of HCQ than free HCQ", "label": "IMPROVEMENT"}}]}}

EXAMPLE 2
ABSTRACT: We propose TinyAlign , a distillation method that transfers reasoning to a smaller student . Evaluated on GLUE and ImageNet , TinyAlign retains 97% of teacher accuracy while reducing inference latency 4x over prior distillation baselines .
OUTPUT: {{"entities": [{{"text": "TinyAlign", "label": "CONTRIBUTION"}}, {{"text": "GLUE", "label": "DATASET"}}, {{"text": "ImageNet", "label": "DATASET"}}, {{"text": "teacher accuracy", "label": "METRIC"}}, {{"text": "reducing inference latency 4x over prior distillation baselines", "label": "IMPROVEMENT"}}]}}

Now annotate this abstract.
ABSTRACT:
{abstract}
"""

HL_SYSTEM = (
    "You write concise research highlights (the key contributions) of a paper "
    "given its abstract. You output only the highlights, no preamble."
)

HL_USER_TMPL = """Write the research highlights for the paper below as 3-5 short declarative sentences, each stating one key contribution or finding. Base them only on the abstract. Output the sentences only, no numbering, no preamble.

ABSTRACT:
{abstract}
"""

NER_PROMPT_v1 = """

You are an expert annotator for scientific-paper NER. Given a paper ABSTRACT, identify the spans that play a research role and label each with exactly one of the nine labels below. The purpose is to build training data for a model that identifies research-related entities in abstracts.

## Labels

- PROBLEM — the gap, limitation, or challenge that motivates the work.
- OBJECTIVE — the stated aim or goal of the work.
- METHOD — an established or applied technique, approach, or procedure that is NOT the paper's novel contribution.
- CONTRIBUTION — the novel artifact the paper introduces (model, algorithm, theorem, framework, system, compound, or newly released dataset).
- FINDING — a result, observation, or something demonstrated/proven, stated WITHOUT an explicit comparison.
- IMPROVEMENT — a result stated as a comparative gain over a baseline, prior work, or alternative.
- DATASET — an existing data resource that is used or referenced (dataset, benchmark, corpus).
- METRIC — a named measured quantity, variable, indicator, or evaluation measure (e.g. "Time to Collision", "F1 score", "crash rate", "standard deviation of lateral position", "response time"). Tag the name of the measure itself.
- KEY_CONCEPT — a salient domain object or technical term central to the work that does not fit any role above (e.g., a mathematical object, a cell line, a biological process, a model component).

## Precedence rules (resolve conflicts deterministically)

1. If a span is the paper's novel proposed thing, use CONTRIBUTION — even when it is also a method or a dataset. (A newly released dataset is CONTRIBUTION, not DATASET.)
2. If a result is explicitly comparative (vs a baseline, prior work, or alternative), use IMPROVEMENT rather than FINDING.
3. If a span names a measured quantity or evaluation measure, use METRIC rather than KEY_CONCEPT.
4. A metric mentioned on its own is METRIC. But when a measured quantity is embedded inside a FINDING, IMPROVEMENT, or OBJECTIVE span (e.g. "a mean Time to Collision of 3.2 s", "reducing inference latency 4x"), keep it as part of that larger span and do NOT also tag METRIC separately — spans must never overlap.
5. Spans must not overlap. Each character belongs to at most one entity. When two labels could apply, pick the single best one using the rules above.

## Span conventions

- Tag the minimal phrase that carries the meaning. Exclude leading articles ("a", "the") and trailing punctuation, unless they are part of a proper name.
- KEY_CONCEPT must be a short noun phrase naming a single object or term (typically 1–4 words). Do NOT tag whole clauses, sentences, findings, or measured quantities as KEY_CONCEPT — use the appropriate role label (FINDING, METRIC, etc.) or leave the text untagged.
- Copy the surface text EXACTLY as it appears in the abstract. Do not paraphrase, normalize, correct, or re-case anything.
- Not every token needs a label. Leave connective and background text untagged. Prefer precision over coverage — only tag spans that clearly carry a research role.
- Tag each distinct mention separately, including repeated mentions and abbreviations (e.g. tag both "TinyAlign" mentions; tag "RT", "HCQ HMSN").
- List entities in strict order of first appearance in the text (left to right).

## Output format

Return ONLY a JSON object — no prose, no explanation, no markdown code fences. Use exactly this schema:

{
  "text": "<the abstract, reproduced exactly as given>",
  "entities": [
    {"text": "<exact surface string>", "label": "<ONE LABEL>"}
  ]
}

Entities MUST be listed in order of appearance. Do NOT include character offsets — they are computed downstream by matching each surface string left to right against "text".

## Example 1

ABSTRACT:
Radiotherapy is a major modality for cancer treatment along with surgery and chemotherapy . Despite its therapeutic effect the recurrence and metastasis of tumors due to the acquired resistance of cancer cells to RT remain significant clinical problems . Therefore it is imperative to overcome radioresistance and improve radiosensitivity in cancer patients . Here we synthesized hydroxychloroquine loaded hollow mesoporous silica nanoparticles to enable effective inhibition of radiation induced cytoprotective autophagy and enhance the therapeutic efficacy of RT . HCQ HMSN treated HCT116 colon cancer cells showed a 200 fold higher intracellular uptake of HCQ than that of free HCQ treated cells thereby effectively inhibiting the radiation induced autophagy of cancer cells .

OUTPUT:
{
  "text": "Radiotherapy is a major modality for cancer treatment along with surgery and chemotherapy . Despite its therapeutic effect the recurrence and metastasis of tumors due to the acquired resistance of cancer cells to RT remain significant clinical problems . Therefore it is imperative to overcome radioresistance and improve radiosensitivity in cancer patients . Here we synthesized hydroxychloroquine loaded hollow mesoporous silica nanoparticles to enable effective inhibition of radiation induced cytoprotective autophagy and enhance the therapeutic efficacy of RT . HCQ HMSN treated HCT116 colon cancer cells showed a 200 fold higher intracellular uptake of HCQ than that of free HCQ treated cells thereby effectively inhibiting the radiation induced autophagy of cancer cells .",
  "entities": [
    {"text": "Radiotherapy", "label": "METHOD"},
    {"text": "cancer", "label": "KEY_CONCEPT"},
    {"text": "surgery", "label": "METHOD"},
    {"text": "chemotherapy", "label": "METHOD"},
    {"text": "recurrence and metastasis of tumors", "label": "PROBLEM"},
    {"text": "acquired resistance of cancer cells to RT", "label": "PROBLEM"},
    {"text": "overcome radioresistance", "label": "OBJECTIVE"},
    {"text": "improve radiosensitivity", "label": "OBJECTIVE"},
    {"text": "hydroxychloroquine loaded hollow mesoporous silica nanoparticles", "label": "CONTRIBUTION"},
    {"text": "radiation induced cytoprotective autophagy", "label": "KEY_CONCEPT"},
    {"text": "enhance the therapeutic efficacy of RT", "label": "OBJECTIVE"},
    {"text": "HCQ HMSN", "label": "CONTRIBUTION"},
    {"text": "HCT116 colon cancer cells", "label": "KEY_CONCEPT"},
    {"text": "200 fold higher intracellular uptake of HCQ", "label": "IMPROVEMENT"},
    {"text": "inhibiting the radiation induced autophagy", "label": "FINDING"}
  ]
}

## Example 2

ABSTRACT:
Knowledge distillation typically requires large labeled corpora that are costly to curate . We propose TinyAlign , a distillation method that transfers reasoning from a teacher to a 10x smaller student . Evaluated on GLUE and ImageNet , TinyAlign retains 97% of teacher accuracy while reducing inference latency 4x over prior distillation baselines .

OUTPUT:
{
  "text": "Knowledge distillation typically requires large labeled corpora that are costly to curate . We propose TinyAlign , a distillation method that transfers reasoning from a teacher to a 10x smaller student . Evaluated on GLUE and ImageNet , TinyAlign retains 97% of teacher accuracy while reducing inference latency 4x over prior distillation baselines .",
  "entities": [
    {"text": "Knowledge distillation", "label": "KEY_CONCEPT"},
    {"text": "large labeled corpora that are costly to curate", "label": "PROBLEM"},
    {"text": "TinyAlign", "label": "CONTRIBUTION"},
    {"text": "transfers reasoning from a teacher to a 10x smaller student", "label": "METHOD"},
    {"text": "GLUE", "label": "DATASET"},
    {"text": "ImageNet", "label": "DATASET"},
    {"text": "TinyAlign", "label": "CONTRIBUTION"},
    {"text": "retains 97% of teacher accuracy", "label": "FINDING"},
    {"text": "reducing inference latency 4x over prior distillation baselines", "label": "IMPROVEMENT"}
  ]
}

## Example 3

ABSTRACT:
Rear end collisions are a major safety concern at signalized intersections . This study evaluated driver braking behaviour using a driving simulator . Time to Collision and Post Encroachment Time were recorded as safety indicators . The proposed warning system reduced hard braking events by 18% compared with the baseline system .

OUTPUT:
{
  "text": "Rear end collisions are a major safety concern at signalized intersections . This study evaluated driver braking behaviour using a driving simulator . Time to Collision and Post Encroachment Time were recorded as safety indicators . The proposed warning system reduced hard braking events by 18% compared with the baseline system .",
  "entities": [
    {"text": "Rear end collisions", "label": "PROBLEM"},
    {"text": "signalized intersections", "label": "KEY_CONCEPT"},
    {"text": "evaluated driver braking behaviour", "label": "OBJECTIVE"},
    {"text": "driving simulator", "label": "METHOD"},
    {"text": "Time to Collision", "label": "METRIC"},
    {"text": "Post Encroachment Time", "label": "METRIC"},
    {"text": "proposed warning system", "label": "CONTRIBUTION"},
    {"text": "reduced hard braking events by 18% compared with the baseline system", "label": "IMPROVEMENT"}
  ]
}

## Now annotate

ABSTRACT:
{{ABSTRACT}}

OUTPUT:

"""

NER_PROMPT_v2 = """

You are an expert annotator for scientific-paper NER. Given a paper ABSTRACT from any scientific domain (physics, biology, medicine, computer science, engineering, etc.), identify the spans that play a research role and label each with exactly one of the nine labels below. The purpose is to build training data for a model that identifies research-related entities in abstracts.

## Labels (domain-agnostic definitions)

- PROBLEM — the gap, limitation, challenge, research question, or motivation that justifies the work. Examples: "object detection performance has plateaued", "acquired resistance of cancer cells", "elevated plasmin in COVID-19 patients", "selective death of motor neurons", "fundamental differences between natural images and medical imaging".

- OBJECTIVE — the stated aim, goal, purpose, or what the paper set out to do or study. Examples: "evaluate driver braking behaviour", "establish an intelligent pest recognition system", "explore task-dependent functional coactivations", "explore implications of radius determination".

- METHOD — an established, known, or standard technique, approach, procedure, or scientific tool that is NOT the paper's novel contribution. Examples: "driving simulator", "hadronic cascade model", "factor analysis", "transfer learning", "nonparametric Bayesian clustering", "Green-Kubo formula". Include domain-specific measurement or experimental techniques.

- CONTRIBUTION — the novel artifact, algorithm, model, system, framework, tool, dataset, or approach the paper introduces or proposes. Examples: "R-CNN", "proposed warning system", "two-step pipeline", "Tiny-YOLOv3 neural network model", "constant-speed-of-sound EOS for quark matter", "hydroxychloroquine loaded hollow mesoporous silica nanoparticles". (A newly released dataset or benchmark is CONTRIBUTION, not DATASET.)

- FINDING — a result, observation, measurement, pattern, or discovery stated WITHOUT an explicit comparison to a baseline. Examples: "the flow coefficients exhibit a particle mass dependence", "retains 97% of teacher accuracy", "iEBE-VISHNU calculations describe the results fairly well", "Lipid metabolism plays a role in motor neuron diseases".

- IMPROVEMENT — a result or performance gain stated as a comparative or quantitative claim over a baseline, prior work, alternative method, or control. Includes expressions like "improves by X%", "better than", "outperforms", "superior", "reduces X by Y%", "higher than", "lower than". Examples: "reduced hard braking events by 18% compared with baseline", "improves mean average precision by 30% relative to previous best", "efficiency increases by 10-30%", "200 fold higher intracellular uptake".

- DATASET — an existing, established, or publicly available data resource, benchmark, corpus, database, or measurement campaign that is used, referenced, or evaluated on. Examples: "GLUE", "ImageNet", "PASCAL VOC", "GW170817", "LHC proton-proton collision data". Do NOT tag a newly released or proposed dataset — use CONTRIBUTION instead.

- METRIC — a named measured quantity, variable, evaluation measure, or indicator being tracked or reported. Includes standard metrics, physical quantities, statistical measures, and domain-specific measurements. Examples: "F1 score", "Time to Collision", "mean average precision (mAP)", "flow coefficients", "efficiency", "transverse momentum", "tidal deformability", "D-dimer", "infection rate". Tag the name or phrase of the measure itself. When a metric appears as part of a FINDING or IMPROVEMENT statement (e.g. "reducing inference latency 4x", "200 fold higher uptake"), it may stay embedded in that larger span — use judgment to avoid artificial fragmentation.

- KEY_CONCEPT — a salient domain object, technical term, theoretical concept, physical quantity, biological entity, or model component central to the work that does not fit any of the roles above. Examples: "quark matter", "Dirac fermions", "hybrid stars", "motor neuron", "furin site in S protein", "hyperfibrinolysis", "structured clusters", "particle mass dependence", "mesons and baryons", "spin density". KEY_CONCEPT should be a noun phrase (typically 1–5 words) naming a single object, entity, or concept — NOT a whole clause, sentence, or finding.

## Precedence rules (resolve conflicts deterministically)

1. If a span is the paper's novel proposed thing, use CONTRIBUTION — even when it is also a method or a dataset. (A newly released dataset is CONTRIBUTION, not DATASET.)
2. If a result is explicitly comparative (vs a baseline, prior work, or alternative), use IMPROVEMENT rather than FINDING.
3. If a span names a measured quantity or evaluation measure, prioritize METRIC over KEY_CONCEPT.
4. A metric mentioned on its own is METRIC. When a measured quantity is embedded inside a FINDING, IMPROVEMENT, or OBJECTIVE span (e.g. "a mean Time to Collision of 3.2 s", "reducing inference latency 4x"), keep it as part of that larger span and do NOT tag METRIC separately — spans must never overlap. Use your judgment to avoid excessive fragmentation.
5. Spans must not overlap. Each character belongs to at most one entity. When two labels could apply, pick the single best one using the rules above.
6. For experimental or measurement data presented in the abstract itself (e.g., "Pb-Pb collisions at 5.02 TeV"), tag as FINDING or METRIC depending on whether it is a result or a named measurement quantity. Do NOT treat raw experimental conditions as requiring separate tags unless they play a distinct role.

## Span conventions

- Tag the minimal phrase that carries the meaning. Exclude leading articles ("a", "the") and trailing punctuation, unless they are part of a proper name or abbreviation.
- KEY_CONCEPT must be a short noun phrase naming a single object or term (typically 1–5 words). Do NOT tag whole clauses, sentences, findings, or measured quantities as KEY_CONCEPT — use the appropriate role label (FINDING, METRIC, etc.) or leave the text untagged.
- Copy the surface text EXACTLY as it appears in the abstract. Do not paraphrase, normalize, correct, re-case, or expand abbreviations.
- Not every token needs a label. Leave connective and background text untagged. Prefer precision over coverage — only tag spans that clearly carry a research role.
- Tag each distinct mention separately, including repeated mentions and abbreviations (e.g., tag both "TinyAlign" mentions; tag "RT", "HCQ", "HMSN" if they refer to distinct concepts or appear multiple times as mentions of the contribution or key entity).
- List entities in strict order of first appearance in the text (left to right).

## Output format

Return ONLY a JSON object — no prose, no explanation, no markdown code fences. Use exactly this schema:

{
  "text": "<the abstract, reproduced exactly as given>",
  "entities": [
    {"text": "<exact surface string>", "label": "<ONE LABEL>"}
  ]
}

Entities MUST be listed in order of appearance. Do NOT include character offsets — they are computed downstream by matching each surface string left to right against "text".

## Example 1 (Physics: particle collisions)

ABSTRACT:
The elliptic ($v_2$), triangular ($v_3$), and quadrangular ($v_4$) flow coefficients of $\pi^{\pm}$, ${\rm K}^{\pm}$, ${\rm p+\overline{p}}$, $\Lambda+\overline{\Lambda}$, ${\rm K}^{\rm 0}_{\rm S}$, and the $\phi$-meson are measured in Pb-Pb collisions at $\sqrt{s_{\rm NN}}=5.02$ TeV. Results obtained with the scalar product method are reported for the rapidity range $\vert y \vert <$ 0.5 as a function of transverse momentum, $p_\text{T}$. For $p_\text{T} < 3$ GeV, the flow coefficients exhibit a particle mass dependence. At intermediate transverse momenta, particles show an approximate grouping according to their type. The flow coefficients are compared to iEBE-VISHNU and MUSIC hydrodynamical calculations.

OUTPUT:
{
  "text": "The elliptic ($v_2$), triangular ($v_3$), and quadrangular ($v_4$) flow coefficients of $\\pi^{\\pm}$, ${\\rm K}^{\\pm}$, ${\\rm p+\\overline{p}}$, $\\Lambda+\\overline{\\Lambda}$, ${\\rm K}^{\\rm 0}_{\\rm S}$, and the $\\phi$-meson are measured in Pb-Pb collisions at $\\sqrt{s_{\\rm NN}}=5.02$ TeV. Results obtained with the scalar product method are reported for the rapidity range $\\vert y \\vert <$ 0.5 as a function of transverse momentum, $p_\\text{T}$. For $p_\\text{T} < 3$ GeV, the flow coefficients exhibit a particle mass dependence. At intermediate transverse momenta, particles show an approximate grouping according to their type. The flow coefficients are compared to iEBE-VISHNU and MUSIC hydrodynamical calculations.",
  "entities": [
    {"text": "flow coefficients", "label": "METRIC"},
    {"text": "Pb-Pb collisions", "label": "METHOD"},
    {"text": "scalar product method", "label": "METHOD"},
    {"text": "transverse momentum", "label": "METRIC"},
    {"text": "particle mass dependence", "label": "FINDING"},
    {"text": "iEBE-VISHNU", "label": "DATASET"},
    {"text": "MUSIC", "label": "DATASET"}
  ]
}

## Example 2 (Biomedical: cancer nanomedicine)

ABSTRACT:
Despite its therapeutic effect, the recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain significant clinical problems. Here we synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced cytoprotective autophagy and enhance therapeutic efficacy. HCQ HMSN-treated HCT116 colon cancer cells showed a 200-fold higher intracellular uptake of HCQ than free HCQ-treated cells, thereby effectively inhibiting radiation-induced autophagy.

OUTPUT:
{
  "text": "Despite its therapeutic effect, the recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain significant clinical problems. Here we synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced cytoprotective autophagy and enhance therapeutic efficacy. HCQ HMSN-treated HCT116 colon cancer cells showed a 200-fold higher intracellular uptake of HCQ than free HCQ-treated cells, thereby effectively inhibiting radiation-induced autophagy.",
  "entities": [
    {"text": "recurrence and metastasis of tumors", "label": "PROBLEM"},
    {"text": "acquired resistance of cancer cells to radiotherapy", "label": "PROBLEM"},
    {"text": "hydroxychloroquine-loaded hollow mesoporous silica nanoparticles", "label": "CONTRIBUTION"},
    {"text": "HCQ HMSN", "label": "CONTRIBUTION"},
    {"text": "inhibit radiation-induced cytoprotective autophagy", "label": "OBJECTIVE"},
    {"text": "enhance therapeutic efficacy", "label": "OBJECTIVE"},
    {"text": "HCT116 colon cancer cells", "label": "KEY_CONCEPT"},
    {"text": "intracellular uptake", "label": "METRIC"},
    {"text": "200-fold higher intracellular uptake of HCQ", "label": "IMPROVEMENT"}
  ]
}

## Example 3 (Computer Vision: deep learning)

ABSTRACT:
Object detection performance on PASCAL VOC has plateaued in recent years. We propose R-CNN, a simple detection algorithm combining two key insights: applying high-capacity CNNs to region proposals for object localization, and using supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012. We provide insight into the learned feature hierarchy.

OUTPUT:
{
  "text": "Object detection performance on PASCAL VOC has plateaued in recent years. We propose R-CNN, a simple detection algorithm combining two key insights: applying high-capacity CNNs to region proposals for object localization, and using supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012. We provide insight into the learned feature hierarchy.",
  "entities": [
    {"text": "object detection performance", "label": "PROBLEM"},
    {"text": "PASCAL VOC", "label": "DATASET"},
    {"text": "R-CNN", "label": "CONTRIBUTION"},
    {"text": "high-capacity CNNs", "label": "METHOD"},
    {"text": "region proposals", "label": "METHOD"},
    {"text": "supervised pre-training", "label": "METHOD"},
    {"text": "domain-specific fine-tuning", "label": "METHOD"},
    {"text": "mean average precision", "label": "METRIC"},
    {"text": "improves mean average precision by more than 30% relative to previous results", "label": "IMPROVEMENT"},
    {"text": "learned feature hierarchy", "label": "FINDING"}
  ]
}

## Example 4 (Remote Sensing / Engineering: pest detection system)

ABSTRACT:
Tessaratoma papillosa causes severe damage to longan crops annually. We developed an intelligent pest recognition system using edge intelligence. A detecting drone photographs pests, and a Tiny-YOLOv3 neural network deployed on NVIDIA Jetson TX2 performs real-time pest recognition. The system transmits pest position data to the cloud for analysis, enabling farmers to understand pest distribution and optimize pesticide spraying, reducing pesticide use and environmental damage.

OUTPUT:
{
  "text": "Tessaratoma papillosa causes severe damage to longan crops annually. We developed an intelligent pest recognition system using edge intelligence. A detecting drone photographs pests, and a Tiny-YOLOv3 neural network deployed on NVIDIA Jetson TX2 performs real-time pest recognition. The system transmits pest position data to the cloud for analysis, enabling farmers to understand pest distribution and optimize pesticide spraying, reducing pesticide use and environmental damage.",
  "entities": [
    {"text": "Tessaratoma papillosa", "label": "KEY_CONCEPT"},
    {"text": "severe damage to longan crops", "label": "PROBLEM"},
    {"text": "intelligent pest recognition system", "label": "CONTRIBUTION"},
    {"text": "edge intelligence", "label": "METHOD"},
    {"text": "detecting drone", "label": "METHOD"},
    {"text": "Tiny-YOLOv3", "label": "CONTRIBUTION"},
    {"text": "NVIDIA Jetson TX2", "label": "METHOD"},
    {"text": "real-time pest recognition", "label": "OBJECTIVE"},
    {"text": "pest distribution", "label": "KEY_CONCEPT"}
  ]
}

## Now annotate

ABSTRACT:
{{ABSTRACT}}

OUTPUT:

"""


NER_PROMPT_v3 = """

You are an expert annotator for scientific-paper NER. Given a paper ABSTRACT from any scientific domain (physics, biology, medicine, computer science, engineering, etc.), identify the spans that play a research role and label each with exactly one of the nine labels below. The purpose is to build training data for a model that identifies research-related entities in abstracts.

## Labels (domain-agnostic definitions)

- PROBLEM — the gap, limitation, challenge, research question, or motivation that justifies the work. Examples: "object detection performance has plateaued", "acquired resistance of cancer cells", "elevated plasmin in COVID-19 patients", "selective death of motor neurons", "fundamental differences between natural images and medical imaging", "short shelf life due to their high moisture content".

- OBJECTIVE — the stated aim, goal, purpose, or what the paper set out to do or study. Examples: "evaluate driver braking behaviour", "establish an intelligent pest recognition system", "explore task-dependent functional coactivations", "estimate and compare cloacal temperature of chicks submitted to thermal stress".

- METHOD — an established, known, or standard technique, approach, procedure, or scientific tool that is NOT the paper's novel contribution. Examples: "driving simulator", "hadronic cascade model", "factor analysis", "transfer learning", "nonparametric Bayesian clustering", "Green-Kubo formula", "cold plasma treatment", "Fuzzy Inference System". Include domain-specific measurement or experimental techniques.

- CONTRIBUTION — the novel artifact, algorithm, model, system, framework, tool, dataset, or approach the paper introduces or proposes. Examples: "R-CNN", "proposed warning system", "two-step pipeline", "Tiny-YOLOv3 neural network model", "constant-speed-of-sound EOS for quark matter", "hydroxychloroquine loaded hollow mesoporous silica nanoparticles", "self assembled nanotheranostics with dual activatable function". (A newly released dataset or benchmark is CONTRIBUTION, not DATASET.)

- FINDING — a result, observation, measurement, pattern, or discovery stated WITHOUT an explicit comparison to a baseline. Examples: "the flow coefficients exhibit a particle mass dependence", "retains 97% of teacher accuracy", "iEBE-VISHNU calculations describe the results fairly well", "Lipid metabolism plays a role in motor neuron diseases", "nanoparticles induce a tighter structure".

- IMPROVEMENT — a result or performance gain stated as a comparative or quantitative claim over a baseline, prior work, alternative method, or control. MUST include BOTH a quantitative measure or comparative operator (improves by X%, better than, higher, lower, % relative to, fold higher) AND an explicit or strongly implied baseline (prior work, baseline, control, alternative). 
  
  ✅ Examples (IMPROVEMENT):
  - "reduced hard braking events by 18% compared with baseline"
  - "improves mean average precision by 30% relative to previous best"
  - "efficiency increases by 10-30%"
  - "200 fold higher intracellular uptake of HCQ than free HCQ-treated cells"
  - "cold plasma induced a lower surface water content"
  
  ❌ Counter-examples (use FINDING instead):
  - "excellent performance" (no quantification, no baseline)
  - "system improves" (vague, no metric)
  - "nanoparticles show enhanced uptake" (qualitative, no comparison stated)

- DATASET — an existing, established, or publicly available data resource, benchmark, corpus, database, or measurement campaign that is used, referenced, or evaluated on. Examples: "GLUE", "ImageNet", "PASCAL VOC", "GW170817", "LHC proton-proton collision data". Do NOT tag a newly released or proposed dataset — use CONTRIBUTION instead.

- METRIC — a named measured quantity, variable, evaluation measure, or indicator being tracked or reported. Includes standard metrics, physical quantities, statistical measures, and domain-specific measurements. Examples: "F1 score", "Time to Collision", "mean average precision (mAP)", "flow coefficients", "efficiency", "transverse momentum", "tidal deformability", "D-dimer", "infection rate", "intracellular uptake", "pH", "concentration", "surface water content". 
  
  CRITICAL: When a metric appears in an IMPROVEMENT statement, INCLUDE the numerical value in the span. Examples:
  - "reduces inference latency 4x" → tag as ONE span (IMPROVEMENT, includes the "4x")
  - "excellent catalytic ability in pH 2–10" → MUST include the pH range in the tagged span
  - "200-fold higher intracellular uptake" → ONE span including "200-fold"
  
  Do NOT extract "excellent catalytic ability" alone (missing the pH 2–10 specificity).

- KEY_CONCEPT — a noun phrase (typically 1–5 words) naming a persistent object, entity, or concept in the domain that is NOT already captured by CONTRIBUTION, DATASET, or METRIC. Salient domain objects, technical terms, theoretical concepts, physical quantities, biological entities, or model components central to the work.
  
  Tagging rule:
  - If it's a *newly proposed* thing → CONTRIBUTION (not KEY_CONCEPT)
  - If it's an *established technique* → METHOD (not KEY_CONCEPT)
  - If it's a *measured quantity or result* → METRIC or FINDING (not KEY_CONCEPT)
  - If it's a *named standard benchmark* → DATASET (not KEY_CONCEPT)
  - Otherwise, it's KEY_CONCEPT — tag it liberally.
  
  Examples by domain:
  - Physics: "quark matter", "Dirac fermions", "particle mass dependence", "flow coefficients" (when used as a concept, not a measurement)
  - Medicine/Biology: "motor neuron", "autophagy", "furin site in S protein", "HCT116 colon cancer cells", "D-dimer", "infection rate"
  - Engineering: "edge intelligence", "pest distribution", "drone", "nanoparticles", "cold plasma"
  - Food Science: "barriques", "oak corks", "sulfites", "gelatin", "wine naturalness"
  - Materials: "cathode materials", "hollow mesoporous silica", "lithium ion batteries"
  
  ACTION: When in doubt, tag KEY_CONCEPT. It is NOT a last resort — it's a common, safe category for domain terms.

## Precedence rules (resolve conflicts deterministically)

1. If a span is the paper's novel proposed thing, use CONTRIBUTION — even when it is also a method or a dataset. (A newly released dataset is CONTRIBUTION, not DATASET.)
2. If a result is explicitly comparative (vs a baseline, prior work, or alternative) AND quantified, use IMPROVEMENT rather than FINDING.
3. If a span names a measured quantity or evaluation measure, prioritize METRIC over KEY_CONCEPT.
4. **OVERLAP RESOLUTION (CLARIFIED):**
   
   (a) If a metric is the *sole focus* of a statement, tag it standalone as METRIC:
       - "flow coefficients exhibit..." → tag "flow coefficients" as METRIC
       - "pH range affects..." → tag "pH range" as METRIC
   
   (b) If a metric is *embedded in a comparative claim (IMPROVEMENT)*, include it in the IMPROVEMENT span:
       - "reducing inference latency 4x" → ONE span "reducing inference latency 4x" as IMPROVEMENT
       - "200-fold higher intracellular uptake of HCQ" → ONE span as IMPROVEMENT
       - Do NOT tag "4x" or "200-fold" separately as METRIC
   
   (c) If a metric *describes or quantifies a KEY_CONCEPT, METHOD, or CONTRIBUTION*, tag separately:
       - "nanoparticles (1300 nm diameter)" → tag "nanoparticles" as CONTRIBUTION, 
         "1300 nm" as METRIC separately (distinct roles, different annotators would mark both)
       - "pH 2–10 range" qualifying a material property → tag both "pH 2–10" (METRIC) and 
         "material property" (KEY_CONCEPT) if they play distinct roles
   
   Rule: Use judgment, but *document* your decision per case. When uncertain, prefer separate tags over consolidation (this helps model learn finer-grained distinctions).

5. Spans must not overlap. Each character belongs to at most one entity. When two labels could apply, pick the single best one using the rules above.

6. For experimental or measurement data presented in the abstract itself (e.g., "Pb-Pb collisions at 5.02 TeV"), tag as FINDING or METRIC depending on whether it is a result or a named measurement quantity. Do NOT treat raw experimental conditions as requiring separate tags unless they play a distinct role.

## Span conventions

- Tag the minimal phrase that carries the meaning. Exclude leading articles ("a", "the") and trailing punctuation, unless they are part of a proper name or abbreviation.
- KEY_CONCEPT must be a short noun phrase naming a single object or term (typically 1–5 words). Do NOT tag whole clauses, sentences, findings, or measured quantities as KEY_CONCEPT — use the appropriate role label (FINDING, METRIC, IMPROVEMENT, etc.) or leave the text untagged.
- **QUANTITATIVE PRESERVATION (CRITICAL):** Many abstracts include numerical specifics (pH ranges, percentages, thresholds, concentrations, fold-changes). These MUST be captured as part of the entity text, not omitted:
  - ✅ "excellent catalytic ability in pH 2–10" (IMPROVEMENT or METHOD, WITH pH range)
  - ❌ "excellent catalytic ability" (incomplete, loses pH specificity)
  - ✅ "improves mean average precision by more than 30%" (IMPROVEMENT, WITH the 30%)
  - ❌ "improves performance" (vague, loses quantification)
  - ✅ "200-fold higher intracellular uptake" (IMPROVEMENT, WITH the fold-change)
  - ❌ "higher uptake" (loses the magnitude)
- Copy the surface text EXACTLY as it appears in the abstract. Do not paraphrase, normalize, correct, re-case, or expand abbreviations.
- Not every token needs a label. Leave connective and background text untagged. Prefer precision over coverage — only tag spans that clearly carry a research role.
- Tag each distinct mention separately, including repeated mentions and abbreviations (e.g., tag both "TinyAlign" mentions; tag "RT", "HCQ", "HMSN" if they refer to distinct concepts or appear multiple times as mentions of the contribution or key entity).
- List entities in strict order of first appearance in the text (left to right).

## Avoid These Common Errors (Based on Model Evaluation)

1. ❌ Under-tagging KEY_CONCEPT: If a domain-specific term (e.g., "motor neurons", "barriques", "cathode materials", "autophagy") is salient to the work, tag it. The model tends to skip domain terms fearing over-tagging. **Be aggressive with KEY_CONCEPT.**

2. ❌ Combining IMPROVEMENT + FINDING improperly: "system reduces toxicity while improving efficacy" — tag these as TWO separate entities if they're distinct claims. "reduces toxicity by 20%" is IMPROVEMENT; "improves efficacy" may be FINDING (if not quantified).

3. ❌ Losing quantitative detail: "achieves 53.3% mAP on VOC 2012" should be tagged to INCLUDE the number. Don't extract just "achieves mAP" (loses the 53.3% specificity).

4. ❌ Vague IMPROVEMENT: "significantly improved" or "remarkably enhanced" lacks quantification. Either quantify it ("improved by 30%") or downgrade to FINDING.

5. ❌ PROBLEM ambiguity: Don't tag generic background motivation ("cancer is a serious disease"). Tag the *specific gap* the paper addresses ("acquired resistance of cancer cells to radiotherapy remains a problem").

6. ❌ Confusing METHOD with CONTRIBUTION: A "Fuzzy Inference System" used in the work is METHOD. A "novel Fuzzy Genetic algorithm" developed by the authors is CONTRIBUTION.

## Output format

Return ONLY a JSON object — no prose, no explanation, no markdown code fences. Use exactly this schema:

{
  "text": "<the abstract, reproduced exactly as given>",
  "entities": [
    {"text": "<exact surface string>", "label": "<ONE LABEL>"}
  ]
}

Entities MUST be listed in order of appearance. Do NOT include character offsets — they are computed downstream by matching each surface string left to right against "text".

## Example 1 (Physics: particle collisions)

ABSTRACT:
The elliptic ($v_2$), triangular ($v_3$), and quadrangular ($v_4$) flow coefficients of $\\pi^{\\pm}$, ${\\rm K}^{\\pm}$, ${\\rm p+\\overline{p}}$, $\\Lambda+\\overline{\\Lambda}$, ${\\rm K}^{\\rm 0}_{\\rm S}$, and the $\\phi$-meson are measured in Pb-Pb collisions at $\\sqrt{s_{\\rm NN}}=5.02$ TeV. Results obtained with the scalar product method are reported for the rapidity range $\\vert y \\vert <$ 0.5 as a function of transverse momentum, $p_\\text{T}$. For $p_\\text{T} < 3$ GeV, the flow coefficients exhibit a particle mass dependence. At intermediate transverse momenta, particles show an approximate grouping according to their type. The flow coefficients are compared to iEBE-VISHNU and MUSIC hydrodynamical calculations.

OUTPUT:
{
  "text": "The elliptic ($v_2$), triangular ($v_3$), and quadrangular ($v_4$) flow coefficients of $\\pi^{\\pm}$, ${\\rm K}^{\\pm}$, ${\\rm p+\\overline{p}}$, $\\Lambda+\\overline{\\Lambda}$, ${\\rm K}^{\\rm 0}_{\\rm S}$, and the $\\phi$-meson are measured in Pb-Pb collisions at $\\sqrt{s_{\\rm NN}}=5.02$ TeV. Results obtained with the scalar product method are reported for the rapidity range $\\vert y \\vert <$ 0.5 as a function of transverse momentum, $p_\\text{T}$. For $p_\\text{T} < 3$ GeV, the flow coefficients exhibit a particle mass dependence. At intermediate transverse momenta, particles show an approximate grouping according to their type. The flow coefficients are compared to iEBE-VISHNU and MUSIC hydrodynamical calculations.",
  "entities": [
    {"text": "flow coefficients", "label": "METRIC"},
    {"text": "Pb-Pb collisions", "label": "METHOD"},
    {"text": "scalar product method", "label": "METHOD"},
    {"text": "transverse momentum", "label": "METRIC"},
    {"text": "particle mass dependence", "label": "FINDING"},
    {"text": "iEBE-VISHNU", "label": "DATASET"},
    {"text": "MUSIC", "label": "DATASET"}
  ]
}

## Example 2 (Biomedical: cancer nanomedicine)

ABSTRACT:
Despite its therapeutic effect, the recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain significant clinical problems. Here we synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced cytoprotective autophagy and enhance therapeutic efficacy. HCQ HMSN-treated HCT116 colon cancer cells showed a 200-fold higher intracellular uptake of HCQ than free HCQ-treated cells, thereby effectively inhibiting radiation-induced autophagy.

OUTPUT:
{
  "text": "Despite its therapeutic effect, the recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain significant clinical problems. Here we synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced cytoprotective autophagy and enhance therapeutic efficacy. HCQ HMSN-treated HCT116 colon cancer cells showed a 200-fold higher intracellular uptake of HCQ than free HCQ-treated cells, thereby effectively inhibiting radiation-induced autophagy.",
  "entities": [
    {"text": "recurrence and metastasis of tumors", "label": "PROBLEM"},
    {"text": "acquired resistance of cancer cells to radiotherapy", "label": "PROBLEM"},
    {"text": "hydroxychloroquine-loaded hollow mesoporous silica nanoparticles", "label": "CONTRIBUTION"},
    {"text": "HCQ HMSN", "label": "CONTRIBUTION"},
    {"text": "inhibit radiation-induced cytoprotective autophagy", "label": "OBJECTIVE"},
    {"text": "enhance therapeutic efficacy", "label": "OBJECTIVE"},
    {"text": "HCT116 colon cancer cells", "label": "KEY_CONCEPT"},
    {"text": "200-fold higher intracellular uptake of HCQ", "label": "IMPROVEMENT"},
    {"text": "radiation-induced autophagy", "label": "KEY_CONCEPT"}
  ]
}

Note: "intracellular uptake" is embedded in the IMPROVEMENT span "200-fold higher intracellular uptake of HCQ" 
to preserve quantitative detail (the "200-fold" is critical). "autophagy" is tagged separately as KEY_CONCEPT 
because it's a salient biological process independent of the uptake claim.

## Example 3 (Computer Vision: deep learning)

ABSTRACT:
Object detection performance on PASCAL VOC has plateaued in recent years. We propose R-CNN, a simple detection algorithm combining two key insights: applying high-capacity CNNs to region proposals for object localization, and using supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012. We provide insight into the learned feature hierarchy.

OUTPUT:
{
  "text": "Object detection performance on PASCAL VOC has plateaued in recent years. We propose R-CNN, a simple detection algorithm combining two key insights: applying high-capacity CNNs to region proposals for object localization, and using supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012. We provide insight into the learned feature hierarchy.",
  "entities": [
    {"text": "object detection performance", "label": "PROBLEM"},
    {"text": "PASCAL VOC", "label": "DATASET"},
    {"text": "R-CNN", "label": "CONTRIBUTION"},
    {"text": "high-capacity CNNs", "label": "METHOD"},
    {"text": "region proposals", "label": "METHOD"},
    {"text": "supervised pre-training", "label": "METHOD"},
    {"text": "domain-specific fine-tuning", "label": "METHOD"},
    {"text": "improves mean average precision (mAP) by more than 30% relative to previous results", "label": "IMPROVEMENT"},
    {"text": "achieving 53.3% mAP on VOC 2012", "label": "IMPROVEMENT"},
    {"text": "learned feature hierarchy", "label": "FINDING"}
  ]
}

Note: The IMPROVEMENT spans INCLUDE the quantitative details (30%, 53.3%) because these numbers are critical 
to the claims and must not be lost. "mean average precision" is subsumed in IMPROVEMENT, not tagged separately as METRIC.

## Example 4 (Materials Science: battery recycling)

ABSTRACT:
Recycling of valuable secondary source of materials from spent batteries brings great opportunity for recyclers at end of life but a challenging issue. Peroxymonosulfate based advanced oxidation has been considered an efficient approach to eliminate refractory antibiotic in wastewater. In this study, cathode materials recovered from spent lithium-ion batteries (LIBs) were used as a catalyst to activate peroxymonosulfate (PMS) for antibiotic degradation. The recovered catalyst material exhibited excellent catalytic ability in pH 2–10, efficiently removing the contaminant. The non-radical and radical processes both governed the PMS activation mechanism.

OUTPUT:
{
  "text": "Recycling of valuable secondary source of materials from spent batteries brings great opportunity for recyclers at end of life but a challenging issue. Peroxymonosulfate based advanced oxidation has been considered an efficient approach to eliminate refractory antibiotic in wastewater. In this study, cathode materials recovered from spent lithium-ion batteries (LIBs) were used as a catalyst to activate peroxymonosulfate (PMS) for antibiotic degradation. The recovered catalyst material exhibited excellent catalytic ability in pH 2–10, efficiently removing the contaminant. The non-radical and radical processes both governed the PMS activation mechanism.",
  "entities": [
    {"text": "recycling of valuable materials from spent batteries", "label": "PROBLEM"},
    {"text": "eliminate refractory antibiotic in wastewater", "label": "OBJECTIVE"},
    {"text": "cathode materials recovered from spent lithium-ion batteries", "label": "CONTRIBUTION"},
    {"text": "catalyst to activate peroxymonosulfate", "label": "CONTRIBUTION"},
    {"text": "peroxymonosulfate based advanced oxidation", "label": "METHOD"},
    {"text": "antibiotic degradation", "label": "OBJECTIVE"},
    {"text": "excellent catalytic ability in pH 2–10", "label": "IMPROVEMENT"},
    {"text": "cathode materials", "label": "KEY_CONCEPT"},
    {"text": "peroxymonosulfate", "label": "KEY_CONCEPT"},
    {"text": "non-radical and radical processes", "label": "FINDING"}
  ]
}

Note: The IMPROVEMENT span "excellent catalytic ability in pH 2–10" INCLUDES the pH range (2–10) because 
this quantitative detail is essential. Without it, the highlight loses critical specificity. "cathode materials" 
and "peroxymonosulfate" are tagged as KEY_CONCEPT because they're salient materials/concepts in the domain.

## Example 5 (Food Science: wine perception study)

ABSTRACT:
Consumers around the globe are showing an increasing demand for food products that are perceived as being natural. The present study aimed to assess what properties of wine have an influence on the perceived naturalness of wine with a focus on winemaking techniques. An online survey was conducted with 500 respondents from Switzerland and Australia. Results showed that wine naturalness was more important to Swiss respondents compared to Australian respondents. An Old World origin, aging in barriques or oak corks were perceived as the most natural properties of wine. The addition of sugar, sulfites, or the use of gelatin were perceived as the most unnatural properties. Additives were perceived as significantly less natural than processing aids and technologies.

OUTPUT:
{
  "text": "Consumers around the globe are showing an increasing demand for food products that are perceived as being natural. The present study aimed to assess what properties of wine have an influence on the perceived naturalness of wine with a focus on winemaking techniques. An online survey was conducted with 500 respondents from Switzerland and Australia. Results showed that wine naturalness was more important to Swiss respondents compared to Australian respondents. An Old World origin, aging in barriques or oak corks were perceived as the most natural properties of wine. The addition of sugar, sulfites, or the use of gelatin were perceived as the most unnatural properties. Additives were perceived as significantly less natural than processing aids and technologies.",
  "entities": [
    {"text": "increasing demand for natural food products", "label": "PROBLEM"},
    {"text": "assess what properties of wine have an influence on perceived naturalness", "label": "OBJECTIVE"},
    {"text": "online survey", "label": "METHOD"},
    {"text": "wine naturalness", "label": "METRIC"},
    {"text": "wine naturalness was more important to Swiss respondents compared to Australian respondents", "label": "IMPROVEMENT"},
    {"text": "Old World origin", "label": "KEY_CONCEPT"},
    {"text": "barriques", "label": "KEY_CONCEPT"},
    {"text": "oak corks", "label": "KEY_CONCEPT"},
    {"text": "sugar", "label": "KEY_CONCEPT"},
    {"text": "sulfites", "label": "KEY_CONCEPT"},
    {"text": "gelatin", "label": "KEY_CONCEPT"},
    {"text": "additives perceived as significantly less natural than processing aids", "label": "IMPROVEMENT"}
  ]
}

Note: Wine-related materials (barriques, oak corks, sulfites, gelatin) are tagged as KEY_CONCEPT because 
they are salient domain objects critical to the study's findings. The comparative claims ("more important to Swiss" 
and "less natural than processing aids") are tagged as IMPROVEMENT with the comparison included.

## Example 6 (Agriculture: thermal stress in poultry)

ABSTRACT:
Behaviour and physiological responses could be an indication of the thermal comfort or discomfort of broiler chicks. This study aimed to estimate and compare cloacal temperature of chicks submitted to thermal stress using two predictive models. A Fuzzy Inference System (FIS) and a Fuzzy Genetic Algorithm (FGA) were developed and compared on a dataset of 200 chicks. Descriptive statistical indices were computed to evaluate improvement in the Fuzzy models. The fuzzy GA exhibited better predicting capacity and statistical results than the FIS with a 15% improvement in accuracy. Thermal challenge may affect chick welfare and they may not recover adequately.

OUTPUT:
{
  "text": "Behaviour and physiological responses could be an indication of the thermal comfort or discomfort of broiler chicks. This study aimed to estimate and compare cloacal temperature of chicks submitted to thermal stress using two predictive models. A Fuzzy Inference System (FIS) and a Fuzzy Genetic Algorithm (FGA) were developed and compared on a dataset of 200 chicks. Descriptive statistical indices were computed to evaluate improvement in the Fuzzy models. The fuzzy GA exhibited better predicting capacity and statistical results than the FIS with a 15% improvement in accuracy. Thermal challenge may affect chick welfare and they may not recover adequately.",
  "entities": [
    {"text": "thermal comfort or discomfort of broiler chicks", "label": "PROBLEM"},
    {"text": "estimate and compare cloacal temperature of chicks submitted to thermal stress", "label": "OBJECTIVE"},
    {"text": "Fuzzy Inference System", "label": "METHOD"},
    {"text": "FIS", "label": "METHOD"},
    {"text": "Fuzzy Genetic Algorithm", "label": "METHOD"},
    {"text": "FGA", "label": "METHOD"},
    {"text": "cloacal temperature", "label": "METRIC"},
    {"text": "fuzzy GA exhibited better predicting capacity and statistical results than the FIS with a 15% improvement in accuracy", "label": "IMPROVEMENT"},
    {"text": "broiler chicks", "label": "KEY_CONCEPT"},
    {"text": "thermal stress", "label": "KEY_CONCEPT"},
    {"text": "chick welfare", "label": "KEY_CONCEPT"}
  ]
}

Note: FIS and FGA are tagged separately as both METHOD (they are established techniques) and CONTRIBUTION 
alternatives are considered. The key IMPROVEMENT compares FGA to FIS with a quantified 15% improvement. 
"broiler chicks", "thermal stress", and "chick welfare" are salient domain concepts tagged as KEY_CONCEPT.

## Now annotate

ABSTRACT:
{{ABSTRACT}}

OUTPUT:

"""

NER_PROMPT_v3_LITE = """

You are an NER annotator for scientific abstracts. Identify research-role spans and label with one of nine labels.

## Labels

- PROBLEM — gap, limitation, challenge, or motivation. Examples: "object detection has plateaued", "acquired resistance of cancer cells", "short shelf life".

- OBJECTIVE — stated aim, goal, or purpose. Examples: "evaluate driver behavior", "establish recognition system", "estimate temperature".

- METHOD — established technique, approach, or tool (NOT novel). Examples: "transfer learning", "Fuzzy Inference System", "cold plasma treatment".

- CONTRIBUTION — novel artifact, algorithm, model, or approach introduced. Examples: "R-CNN", "nanotheranostics", "proposed system".

- FINDING — result, observation, or discovery WITHOUT explicit comparison. Examples: "exhibits particle mass dependence", "induces tighter structure".

- IMPROVEMENT — result with BOTH quantification AND baseline comparison. 
  ✅ "reduces events by 18% vs. baseline", "improves by 30%", "200-fold higher"
  ❌ "excellent performance", "system improves" (use FINDING instead)

- DATASET — existing, public data resource or benchmark. Examples: "GLUE", "ImageNet", "PASCAL VOC".

- METRIC — named measured quantity or indicator. Examples: "F1 score", "pH", "intracellular uptake", "efficiency".
  CRITICAL: Include numbers in IMPROVEMENT spans: "reduces latency 4x" (ONE span), "pH 2–10" (preserve range).

- KEY_CONCEPT — salient domain term/concept NOT in other labels. Examples: "motor neuron", "barriques", "cathode materials".
  Tagging rule: If novel→CONTRIBUTION, if technique→METHOD, if measured→METRIC, else→KEY_CONCEPT. **Tag liberally.**

## Key Rules

1. **OVERLAP:** 
   - (a) Standalone metric → tag as METRIC: "flow coefficients exhibit..."
   - (b) Metric in IMPROVEMENT → embed in span: "200-fold higher uptake" (ONE span)
   - (c) Metric describing another label → tag separately: "nanoparticles (1300 nm)"

2. **IMPROVEMENT requires both:** quantification ("30%", "4x", "lower") AND baseline ("vs.", "compared to", "than").

3. **QUANTITATIVE PRESERVATION:** Numbers MUST stay in spans:
   - ✅ "excellent catalytic ability in pH 2–10"
   - ❌ "excellent catalytic ability" (loses pH)

4. **KEY_CONCEPT:** Domain objects are safe—tag them aggressively. Do NOT tag findings/clauses as KEY_CONCEPT.

5. Spans don't overlap. Tag minimal phrase. List in order of appearance. Copy surface text exactly.

## Common Errors to Avoid

1. ❌ Under-tagging KEY_CONCEPT → Be aggressive with domain terms
2. ❌ Losing quantitative detail → Include numbers in spans
3. ❌ Vague IMPROVEMENT → Require quantification or use FINDING
4. ❌ Confusing METHOD vs. CONTRIBUTION → "Fuzzy Inference System" (used) is METHOD; "novel FGA" (developed) is CONTRIBUTION

## Output format

Return ONLY JSON:
{
  "text": "<abstract, exact>",
  "entities": [
    {"text": "<surface string>", "label": "<LABEL>"}
  ]
}

Entities in order of appearance. No offsets.

## Example 1 (Biomedical: cancer nanomedicine)

ABSTRACT:
Despite therapeutic effect, recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain clinical problems. We synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced autophagy and enhance efficacy. HCQ HMSN-treated HCT116 cells showed 200-fold higher intracellular uptake of HCQ than free HCQ cells, effectively inhibiting autophagy.

OUTPUT:
{
  "text": "Despite therapeutic effect, recurrence and metastasis of tumors due to acquired resistance of cancer cells to radiotherapy remain clinical problems. We synthesized hydroxychloroquine-loaded hollow mesoporous silica nanoparticles (HCQ HMSN) to inhibit radiation-induced autophagy and enhance efficacy. HCQ HMSN-treated HCT116 cells showed 200-fold higher intracellular uptake of HCQ than free HCQ cells, effectively inhibiting autophagy.",
  "entities": [
    {"text": "recurrence and metastasis of tumors", "label": "PROBLEM"},
    {"text": "acquired resistance of cancer cells to radiotherapy", "label": "PROBLEM"},
    {"text": "hydroxychloroquine-loaded hollow mesoporous silica nanoparticles", "label": "CONTRIBUTION"},
    {"text": "HCQ HMSN", "label": "CONTRIBUTION"},
    {"text": "inhibit radiation-induced autophagy", "label": "OBJECTIVE"},
    {"text": "enhance efficacy", "label": "OBJECTIVE"},
    {"text": "HCT116 cells", "label": "KEY_CONCEPT"},
    {"text": "200-fold higher intracellular uptake of HCQ", "label": "IMPROVEMENT"},
    {"text": "autophagy", "label": "KEY_CONCEPT"}
  ]
}

## Example 2 (Computer Vision: R-CNN)

ABSTRACT:
Object detection performance on PASCAL VOC has plateaued. We propose R-CNN, combining CNNs with region proposals and supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012.

OUTPUT:
{
  "text": "Object detection performance on PASCAL VOC has plateaued. We propose R-CNN, combining CNNs with region proposals and supervised pre-training followed by domain-specific fine-tuning. Our approach improves mean average precision (mAP) by more than 30% relative to previous results, achieving 53.3% mAP on VOC 2012.",
  "entities": [
    {"text": "object detection performance", "label": "PROBLEM"},
    {"text": "PASCAL VOC", "label": "DATASET"},
    {"text": "R-CNN", "label": "CONTRIBUTION"},
    {"text": "CNNs", "label": "METHOD"},
    {"text": "region proposals", "label": "METHOD"},
    {"text": "supervised pre-training", "label": "METHOD"},
    {"text": "domain-specific fine-tuning", "label": "METHOD"},
    {"text": "improves mean average precision (mAP) by more than 30% relative to previous results", "label": "IMPROVEMENT"},
    {"text": "achieving 53.3% mAP on VOC 2012", "label": "IMPROVEMENT"}
  ]
}

## Example 3 (Materials: Battery recycling)

ABSTRACT:
Cathode materials recovered from spent lithium-ion batteries were used as catalyst to activate peroxymonosulfate for antibiotic degradation. The catalyst exhibited excellent catalytic ability in pH 2–10, efficiently removing contaminant. Non-radical and radical processes both governed PMS activation.

OUTPUT:
{
  "text": "Cathode materials recovered from spent lithium-ion batteries were used as catalyst to activate peroxymonosulfate for antibiotic degradation. The catalyst exhibited excellent catalytic ability in pH 2–10, efficiently removing contaminant. Non-radical and radical processes both governed PMS activation.",
  "entities": [
    {"text": "recycling cathode materials from batteries", "label": "PROBLEM"},
    {"text": "eliminate antibiotic in wastewater", "label": "OBJECTIVE"},
    {"text": "cathode materials recovered from spent lithium-ion batteries", "label": "CONTRIBUTION"},
    {"text": "activate peroxymonosulfate", "label": "CONTRIBUTION"},
    {"text": "peroxymonosulfate based oxidation", "label": "METHOD"},
    {"text": "excellent catalytic ability in pH 2–10", "label": "IMPROVEMENT"},
    {"text": "cathode materials", "label": "KEY_CONCEPT"},
    {"text": "peroxymonosulfate", "label": "KEY_CONCEPT"},
    {"text": "non-radical and radical processes", "label": "FINDING"}
  ]
}

## Now annotate

ABSTRACT:
{{ABSTRACT}}

OUTPUT:

"""


def build_ner_user_prompt(abstract, labels=LABELS):
    """Format the detailed SLM NER instruction for one abstract."""
    return NER_USER_TMPL.format(
        labels_csv=", ".join(labels),
        abstract=abstract,
    )


# ============================================================================
# Type Classification Prompts (Zero-shot and Fine-tuned)
# ============================================================================

TYPE_SYSTEM_PROMPT = """You are an expert at classifying research contributions by type.
Given a CONTRIBUTION span extracted from a paper abstract, determine what kind of contribution it is.

Classification is based on the primary artifact or innovation introduced:
- algorithm: New computational method, neural architecture, optimization procedure, algorithm
- dataset: Dataset creation, benchmark release, corpus, data collection
- theory: Theoretical result, proof, mathematical analysis, characterization
- experiment: Experimental measurement, new experimental procedure, discovery
- framework: Conceptual framework, high-level approach, theoretical framework
- generalization: Extension or generalization of existing work
- software: Software tool, implementation, code package, library

Output the type as a single word."""

TYPE_USER_PROMPT_TEMPLATE = """Classify the contribution type:

"{contribution_text}"

Answer with ONE word only: algorithm, dataset, theory, experiment, framework, generalization, or software.

Type: """

CONTRIBUTION_TYPES = (
    "algorithm",
    "dataset",
    "theory",
    "experiment",
    "framework",
    "generalization",
    "software",
)

# ============================================================================
# NARRATIVE EXTRACTION v1 (3 labels): TASK, LIMITATION, CONTRIBUTION
# ============================================================================

NARRATIVE_SYSTEM_v1 = (
    "You are an expert at decomposing research papers into structured narrative components. "
    "Given an abstract, you extract three key research narrative elements and output them as JSON."
)

NARRATIVE_USER_TEMPLATE_v1 = """Extract the three core narrative components from this research abstract:

1. TASK — the research area, problem domain, or technical topic this paper addresses
2. LIMITATION — the gap, limitation, or unresolved challenge in existing work that motivates this paper
3. CONTRIBUTION — the novel thing proposed: algorithm, framework, approach, tool, analysis, or insight

Extract only what is explicitly stated in the abstract. Keep extractions brief and natural.

Return ONLY valid JSON with no prose, no markdown, no code fences:
{{"TASK": "<extracted task>", "LIMITATION": "<extracted limitation>", "CONTRIBUTION": "<extracted contribution>"}}

## EXAMPLE 1 (Multimodal Vision-Language)

ABSTRACT: The success of large language models (LLMs) has inspired multimodal learning. However, pre-trained LLMs with billions of parameters are difficult to fine-tune efficiently for vision tasks. Models like MiniGPT-4 and LLaVA have been developed but remain limited in artistic image understanding. We propose ArtGPT-4, a large vision-language model tailored for artistic comprehension. It integrates specialized adapter layers into the LLM for efficient artistic image parsing. Trained on 0.52M image-text pairs in 2 hours on a Tesla A100, ArtGPT-4 achieves state-of-the-art on ArtEmis and ArtEmis-v2.0 datasets.

OUTPUT:
{{"TASK": "multimodal vision-language models for artistic image comprehension", "LIMITATION": "existing models like MiniGPT-4 and LLaVA are limited in understanding artistic imagery and require fine-tuning the entire LLM", "CONTRIBUTION": "ArtGPT-4: a vision-language model with specialized adapter layers for efficient artistic image understanding, trainable in 2 hours on 0.52M image-text pairs"}}

## EXAMPLE 2 (Vision Transformers)

ABSTRACT: Transformer architecture is standard for NLP but limited in computer vision, typically used with CNNs. We show that pure transformers applied directly to image patches can perform well on image classification. Vision Transformer (ViT) pre-trained on large datasets and transferred to ImageNet, CIFAR-100, VTAB achieves excellent results compared to state-of-the-art CNNs while requiring fewer computational resources to train.

OUTPUT:
{{"TASK": "applying transformer architectures directly to image classification", "LIMITATION": "transformers in vision are typically combined with CNNs; their standalone effectiveness on images is unclear", "CONTRIBUTION": "Vision Transformer (ViT): a pure transformer architecture applied directly to image patch sequences that achieves competitive results with fewer computational resources than CNN-based approaches"}}

## EXAMPLE 3 (Physics - Entropy Functions)

ABSTRACT: Multiple entropy functions (Bekenstein-Hawking, Tsallis, Rényi, Barrow, Sharma-Mittal, Kaniadakis, Loop Quantum Gravity) are proposed for black hole thermodynamics and cosmology, but no unified framework exists. We propose a new 4-parameter generalized entropy function that converges to known entropies in limiting cases. Extended to non-singular cosmology, it provides singular-free evolution and fundamental understanding of entropy. We address implications for black hole thermodynamics and cosmology with various entropic parameter constraints.

OUTPUT:
{{"TASK": "generalized entropy functions for black hole thermodynamics and cosmology", "LIMITATION": "multiple disparate entropy functions (Bekenstein-Hawking, Tsallis, Rényi, etc.) exist but no unified generalized framework connects them", "CONTRIBUTION": "a 4-parameter generalized entropy function that unifies known entropies in limiting cases, extended to non-singular cosmology for singular-free evolution"}}

Now annotate this abstract:

ABSTRACT:
{abstract}
"""

NARRATIVE_LABELS_v1 = ("TASK", "LIMITATION", "CONTRIBUTION")
# ============================================================================
# NARRATIVE EXTRACTION v2 (5 labels): TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT
# ============================================================================

NARRATIVE_SYSTEM_v2 = (
    "You are an expert at decomposing research papers into structured narrative components. "
    "Given an abstract, you extract five key research narrative elements and output them as JSON."
)

NARRATIVE_USER_TEMPLATE_v2 = """Extract the five core narrative components from this research abstract:

1. TASK — the research area, problem domain, or technical topic this paper addresses
2. LIMITATION — the gap, limitation, or unresolved challenge in existing work that motivates this paper
3. CONTRIBUTION — WHAT the paper adds: a one-line headline naming it and its kind (method/model, dataset/benchmark, software/tool, theoretical result, analysis/study, review/survey, or insight). Every paper has one: if nothing new is built, the contribution is the study or analysis itself (e.g. "a phenomenological analysis of X at the LHC")
4. METHOD — HOW the work is done: techniques, experimental or theoretical approach, data, tools, and settings used (including existing ones the paper relies on)
5. RESULT — WHAT was found: measured outcomes, observed findings, or proven/demonstrated conclusions

Field roles:
- Keep each field focused on its role. CONTRIBUTION names the new thing; METHOD describes how it works or how it was obtained; RESULT reports outcomes. Do not copy the same phrase into several fields.
- Numbers, benchmark scores, and comparisons with prior work belong in RESULT, not CONTRIBUTION.
- TASK and CONTRIBUTION are always non-empty. Only LIMITATION, METHOD and RESULT may be "" when the abstract does not state them. Do not invent content.
- Review/survey and software/tool papers often state no METHOD or RESULT; leave those fields "" rather than filling them with the CONTRIBUTION.

Extract only what is explicitly stated in the abstract. Keep extractions brief and natural.

Return ONLY valid JSON with no prose, no markdown, no code fences:
{{"TASK": "<extracted task>", "LIMITATION": "<extracted limitation>", "CONTRIBUTION": "<extracted contribution>", "METHOD": "<extracted method>", "RESULT": "<extracted result>"}}

## EXAMPLE 1 (Multimodal Vision-Language)

ABSTRACT: The success of large language models (LLMs) has inspired multimodal learning. However, pre-trained LLMs with billions of parameters are difficult to fine-tune efficiently for vision tasks. Models like MiniGPT-4 and LLaVA have been developed but remain limited in artistic image understanding. We propose ArtGPT-4, a large vision-language model tailored for artistic comprehension. It integrates specialized adapter layers into the LLM for efficient artistic image parsing. Trained on 0.52M image-text pairs in 2 hours on a Tesla A100, ArtGPT-4 achieves state-of-the-art on ArtEmis and ArtEmis-v2.0 datasets.

OUTPUT:
{{"TASK": "multimodal vision-language models for artistic image comprehension", "LIMITATION": "existing models like MiniGPT-4 and LLaVA are limited in understanding artistic imagery and require fine-tuning the entire LLM", "CONTRIBUTION": "ArtGPT-4: a large vision-language model tailored for artistic image comprehension", "METHOD": "specialized adapter layers integrated into a pre-trained LLM, trained on 0.52M image-text pairs in 2 hours on a single Tesla A100", "RESULT": "state-of-the-art performance on the ArtEmis and ArtEmis-v2.0 datasets"}}

## EXAMPLE 2 (Vision Transformers)

ABSTRACT: Transformer architecture is standard for NLP but limited in computer vision, typically used with CNNs. We show that pure transformers applied directly to image patches can perform well on image classification. Vision Transformer (ViT) pre-trained on large datasets and transferred to ImageNet, CIFAR-100, VTAB achieves excellent results compared to state-of-the-art CNNs while requiring fewer computational resources to train.

OUTPUT:
{{"TASK": "applying transformer architectures directly to image classification", "LIMITATION": "transformers in vision are typically combined with CNNs; their standalone effectiveness on images is unclear", "CONTRIBUTION": "Vision Transformer (ViT): showing that a pure transformer without CNNs performs well on image classification", "METHOD": "a pure transformer applied to sequences of image patches, pre-trained on large datasets and transferred to ImageNet, CIFAR-100 and VTAB", "RESULT": "excellent results compared to state-of-the-art CNNs while requiring fewer computational resources to train"}}

## EXAMPLE 3 (Physics - Entropy Functions)

ABSTRACT: Multiple entropy functions (Bekenstein-Hawking, Tsallis, Rényi, Barrow, Sharma-Mittal, Kaniadakis, Loop Quantum Gravity) are proposed for black hole thermodynamics and cosmology, but no unified framework exists. We propose a new 4-parameter generalized entropy function that converges to known entropies in limiting cases. Extended to non-singular cosmology, it provides singular-free evolution and fundamental understanding of entropy. We address implications for black hole thermodynamics and cosmology with various entropic parameter constraints.

OUTPUT:
{{"TASK": "generalized entropy functions for black hole thermodynamics and cosmology", "LIMITATION": "multiple disparate entropy functions (Bekenstein-Hawking, Tsallis, Rényi, etc.) exist but no unified generalized framework connects them", "CONTRIBUTION": "a 4-parameter generalized entropy function unifying the known entropy functions", "METHOD": "analytical construction of the generalized entropy, checked against known entropies in limiting cases and extended to non-singular cosmology", "RESULT": "the function reduces to the known entropies in limiting cases and yields singular-free cosmological evolution, with constraints on the entropic parameters"}}

Now annotate this abstract:

ABSTRACT:
{abstract}
"""

NARRATIVE_LABELS_v2 = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")

# Narrative version registry, shared by data generation and inference.
NARRATIVE_VERSIONS = {
    "v1": {
        "system": NARRATIVE_SYSTEM_v1,
        "template": NARRATIVE_USER_TEMPLATE_v1,
        "labels": NARRATIVE_LABELS_v1,
        "description": "3-label narrative extraction: TASK, LIMITATION, CONTRIBUTION",
        "optional": set(),
    },
    "v2": {
        "system": NARRATIVE_SYSTEM_v2,
        "template": NARRATIVE_USER_TEMPLATE_v2,
        "labels": NARRATIVE_LABELS_v2,
        "description": "5-label narrative extraction: TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT",
        "optional": {"LIMITATION", "METHOD", "RESULT"},
    },
}

# Single source of truth for the current (v2) aspect/narrative label set --
# downstream scripts (utils/generate_report.py, utils/facet_scoring.py,
# prediction/qwen_embedding.py) import THIS instead of hand-typing their own
# copy of the tuple, so a hardcoded list can't silently drift out of sync
# with the real schema the way utils/generate_report.py's ASPECT_KEYS once did
# (it was missing RESULT entirely).
ASPECT_KEYS = NARRATIVE_VERSIONS["v2"]["labels"]

# ============================================================================
# LAYER 2: CITATION RELATION FUNCTION (v1)
# ============================================================================
# Purpose: label ONE citation context of reference R inside main paper M with the
#          relation function R plays for M, plus the aspect of M it serves.
# Unit:    one citation context (a reference cited 3 times -> 3 calls).
# Inputs:  M and R titles + 5-label narratives (NARRATIVE_*_v2 output) + context.
#          Render each narrative with LAYER_2_NARRATIVE_BLOCK_v1; replace empty
#          fields with "(not stated)". Join before/after sentence lists with " ".
#          co_cited = "yes" if other references are cited in the same sentence.
#          Strip leftover {{cite:...}} markers from section names; "" -> "(none)".
# Output:  JSON with FUNCTION, M_ASPECT, POLARITY, EVIDENCE, R_MATCH, CONFIDENCE.
# Prompt layout: fixed instructions + examples first, per-citation input last
#          (keeps a long shared prefix for DeepSeek context caching).
# COMPARISON merges compare + contrast; POLARITY tells which (SIMILAR/DIFFERENT).

LAYER_2_RELATION_FUNCTIONS_v1 = (
    "BACKGROUND", "MOTIVATION", "USES", "EXTENDS", "SUPPORTS", "COMPARISON",
)
LAYER_2_M_ASPECTS_v1 = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")
LAYER_2_POLARITIES_v1 = ("SIMILAR", "DIFFERENT", "")
LAYER_2_CONFIDENCE_v1 = ("high", "medium", "low")
LAYER_2_R_MATCH_v1 = ("yes", "no")

LAYER_2_NARRATIVE_BLOCK_v1 = """TASK: {TASK}
LIMITATION: {LIMITATION}
CONTRIBUTION: {CONTRIBUTION}
METHOD: {METHOD}
RESULT: {RESULT}"""

LAYER_2_RELATION_SYSTEM_v1 = (
    "You are an expert in scholarly citation analysis. Given a main paper M, a cited paper R, "
    "and one passage of M where R is cited, you decide what role R plays for M and output it as JSON."
)

LAYER_2_RELATION_USER_TEMPLATE_v1 = """Classify why the MAIN PAPER (M) cites the CITED PAPER (R) in the citation context below.

FUNCTION — the role R plays for M in this context (choose exactly one):
1. BACKGROUND — R is general context for the field or problem; M does not depend on R.
2. MOTIVATION — a gap, limitation, open problem, or finding in R motivates M's work.
   Also MOTIVATION when M describes R's approach in order to point out the problem M addresses (often followed by "however", "issue", "suffers from").
3. USES — M adopts something from R, as it is or adapted into a component of M's own approach: method, model, algorithm, dataset, benchmark, evaluation protocol, software, tool, measured value, or setting (cues: "we use", "we employ", "following", "based on", "inspired by", "we relied on these data").
4. EXTENDS — R's contribution itself is the object M improves, generalizes, or updates, so that M's contribution is a new version of R's; includes M's own earlier work (cues: "we extend", "we update our previous", "we generalize").
5. SUPPORTS — R's result or argument is evidence for a claim, assumption, or design choice of M that M does not test itself, or removes a concern about M's approach (cues: "has been shown to", "as demonstrated in", "has been validated").
6. COMPARISON — M sets its own approach, data, or results against R, including using R as a baseline (cues: "baseline", "consistent with", "similar to", "unlike", "complementary", "outperforms"). Set POLARITY:
   - SIMILAR: agrees, is consistent with, identical or analogous to R
   - DIFFERENT: differs from, is complementary to, outperforms, or disagrees with R

M_ASPECT — which part of M this citation serves (choose exactly one): TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT.
Typical pairs: BACKGROUND→TASK, MOTIVATION→LIMITATION or TASK, USES→METHOD, EXTENDS→CONTRIBUTION or METHOD, SUPPORTS→METHOD or LIMITATION, COMPARISON→RESULT or CONTRIBUTION. Deviate when the context clearly says otherwise.

Rules:
- Decide FUNCTION from the citation context: the citing sentence first, then the sentences before and after. Use the narratives of M and R only to choose M_ASPECT and to resolve ambiguity.
- [Citation] marks where R is cited. When it stands right after a full stop it belongs to the preceding sentence; when it starts a sentence it stands for R's authors (e.g. "[Citation] proposed ...").
- Other citations in the passage have been removed, so some sentences may lack a subject (e.g. "has shown that ...") or contain empty lists. Ignore these gaps.
- Mathematical notation, equations, and "REF" placeholders are extraction residue; read around them.
- If R is co-cited with other references in the same sentence, label the role of R only; use R's narrative to decide what R contributes to that sentence.
- If several functions fit, choose the most specific one, in this order: EXTENDS > USES > SUPPORTS > COMPARISON > MOTIVATION > BACKGROUND.
- POLARITY is "" for every FUNCTION except COMPARISON.
- EVIDENCE is the shortest phrase (at most 12 words) copied verbatim from the context that signals the FUNCTION.
- R_MATCH is "no" if what the context says about the cited work is clearly unrelated to R's narrative (a wrong reference), otherwise "yes". If R_MATCH is "no", still fill the other fields from the context alone.
- CONFIDENCE is "low" if the context is uninformative (e.g. mostly an equation or a heading), the function is a guess, or R_MATCH is "no"; "medium" if two functions fit about equally; otherwise "high".

Return ONLY valid JSON with no prose, no markdown, no code fences:
{{"FUNCTION": "<function>", "M_ASPECT": "<aspect>", "POLARITY": "<SIMILAR|DIFFERENT|empty>", "EVIDENCE": "<phrase from context>", "R_MATCH": "<yes|no>", "CONFIDENCE": "<high|medium|low>"}}

## EXAMPLE 1 (USES)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): Adam: A Method for Stochastic Optimization
TASK: first-order gradient-based optimization of stochastic objective functions
LIMITATION: (not stated)
CONTRIBUTION: Adam: an algorithm for first-order gradient-based optimization of stochastic objective functions based on adaptive estimates of lower-order moments
METHOD: adaptive estimates of lower-order moments, invariant to diagonal rescaling of the gradients
RESULT: computationally efficient, little memory, well suited to noisy and/or sparse gradients

CITATION CONTEXT
Section: Model and Training Details
Co-cited in the same sentence: no
Before:
Citing sentence: All our models are trained using Adam [Citation] for 5,000 epochs, where each epoch is over a chunk of 10,000 training samples.
After: Since we are generating the data, we vary the training samples from 1k to 1M, and use a validation set of 10k samples.

OUTPUT:
{{"FUNCTION": "USES", "M_ASPECT": "METHOD", "POLARITY": "", "EVIDENCE": "All our models are trained using Adam", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 2 (COMPARISON, DIFFERENT)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning
TASK: human-in-the-loop grounded language learning for artificial agents
LIMITATION: current learning methods are poorly data-efficient, making interactive human training of agents impractical
CONTRIBUTION: BabyAI: an extensible research platform for investigating human-in-the-loop grounded language learning
METHOD: an extensible suite of 19 levels of increasing difficulty and a heuristic expert agent simulating a human teacher
RESULT: current deep learning methods are not yet sample-efficient for learning a language with compositional properties

CITATION CONTEXT
Section: Related Work
Co-cited in the same sentence: no
Before: Since proposed, all bAbI tasks have been solved by novel memory architectures.
Citing sentence: An gridworld environment for embodied agents with language instructions for tasks is described in [Citation]; our work here is complementary, giving question-answer pairs based on abstracted environment histories.
After: CLEVR is a popular synthetic data for testing visual reasoning given text queries.

OUTPUT:
{{"FUNCTION": "COMPARISON", "M_ASPECT": "CONTRIBUTION", "POLARITY": "DIFFERENT", "EVIDENCE": "our work here is complementary", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 3 (BACKGROUND, co-cited)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): Attention Is All You Need
TASK: sequence transduction models for machine translation
LIMITATION: the best performing models use complex recurrent or convolutional networks, which are computationally expensive and hard to parallelize
CONTRIBUTION: the Transformer: a new simple network architecture based solely on attention mechanisms
METHOD: attention-based encoder-decoder architecture trained on WMT 2014 translation tasks
RESULT: 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on English-to-French

CITATION CONTEXT
Section: Introduction
Co-cited in the same sentence: yes
Before:
Citing sentence: Advances in machine learning (ML) architectures [Citation], large datasets and model scaling for pre-training, modeling approaches, and multiple dedicated reasoning datasets have driven progress in both building models that can succeed in aspects of “reasoning” and in automatically evaluating such capabilities.
After: This has been evident particularly in the text setting, but also in computer vision.

OUTPUT:
{{"FUNCTION": "BACKGROUND", "M_ASPECT": "TASK", "POLARITY": "", "EVIDENCE": "Advances in machine learning (ML) architectures", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 4 (SUPPORTS)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): TAPEX: Table Pre-training via Learning a Neural SQL Executor
TASK: table pre-training for natural language processing
LIMITATION: large-scale high-quality tabular data is scarce, hindering language model pre-training on structured tabular data
CONTRIBUTION: TAPEX: a table pre-training approach that learns a neural SQL executor over a synthetic corpus
METHOD: guiding a language model to mimic a SQL executor on synthesized executable SQL queries and their outputs
RESULT: new state-of-the-art results on WikiSQL, WikiTableQuestions, SQA and TabFact

CITATION CONTEXT
Section: Environment
Co-cited in the same sentence: no
Before: Multiple time snapshots of the context are represented by appending each successive event in the sequence.
Citing sentence: While this is in some sense simplistic, it allows ready application of large pre-trained language models for ingesting the context; and this kind of approach has been shown to be effective in [Citation].
After: Alternatively, we can leverage the relational properties of the context by representing it as a graph where objects and properties are nodes, and the connections between them are the edges.

OUTPUT:
{{"FUNCTION": "SUPPORTS", "M_ASPECT": "METHOD", "POLARITY": "", "EVIDENCE": "this kind of approach has been shown to be effective", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 5 (COMPARISON, SIMILAR — physics)

MAIN PAPER (M): Columnar magnetic structure coupled with orthorhombic distortion in the antiferromagnetic iron arsenide SrFe2As2
TASK: determining the magnetic structure and its relationship with the crystallographic structure in SrFe2As2
LIMITATION: (not stated)
CONTRIBUTION: a determination of the magnetic propagation vector, interlayer coupling, and magnetic moment of SrFe2As2
METHOD: neutron powder diffraction on polycrystalline SrFe2As2 below T0 = 205 K, with a detailed Rietveld analysis
RESULT: propagation vector q = (1 0 1); antiferromagnetic coupling along the a axis within the Fe-As layer and between layers; moment 1.01(3) μB parallel to the a-axis

CITED PAPER (R): Magnetic order in BaFe2As2, the parent compound of the FeAs based superconductors in a new structural family
TASK: magnetic order in AFe2As2 superconductor parent compounds
LIMITATION: (not stated)
CONTRIBUTION: neutron diffraction studies of BaFe2As2 demonstrating an antiferromagnetic transition concurrent with a first-order structural transition
METHOD: neutron diffraction studies of BaFe2As2
RESULT: an antiferromagnetic transition concurs with a first-order structural transition

CITATION CONTEXT
Section: (none)
Co-cited in the same sentence: yes
Before: The determined magnetic structure of SrFe_2As_{{2}} is shown in Fig. REF; the AFM coupling occurs along the longer a direction and the moment orients in the same direction.
Citing sentence: The magnetic ordering in SrFe_2As_{{2}} within the FeAs layer seems identical to that in BaFe_2As_{{2}} [Citation], in LaFeAs(O,F) and in CeFeAs(O,F), although the stacking order is not common.
After: This supports the importance of the FeAs intralayer coupling as well as the weak interlayer coupling.

OUTPUT:
{{"FUNCTION": "COMPARISON", "M_ASPECT": "RESULT", "POLARITY": "SIMILAR", "EVIDENCE": "seems identical to that in BaFe_2As_{{2}}", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 6 (MOTIVATION — narrative decides M_ASPECT)

MAIN PAPER (M): AutoLoRa: A Parameter-Free Automated Robust Fine-Tuning Framework
TASK: obtaining adversarial robustness through Robust Fine-Tuning (RFT) in deep neural networks
LIMITATION: existing RFT optimizes both adversarial and natural objectives through the feature extractor (FE), yielding divergent gradient directions that introduce instability and sensitivity to hyperparameters
CONTRIBUTION: AutoLoRa: a low-rank (LoRa) branch that disentangles RFT into two distinct components and automates the scheduling of learning rates and loss term scalars
METHOD: a low-rank branch disentangling RFT into optimizing natural objectives via the LoRa branch and adversarial objectives via the FE, with heuristic scheduling of learning rates and loss term scalars
RESULT: new state-of-the-art results across a range of downstream tasks without searching for hyperparameters

CITED PAPER (R): Adversarially robust transfer learning
TASK: robust transfer learning for producing adversarially robust neural network classifiers with scarce data
LIMITATION: data scarcity and computational limitations make it hard to produce both accurate and robust models, and adversarial training is costly
CONTRIBUTION: a method for robust transfer learning that transfers robustness from a source model to a target domain
METHOD: training classifiers on top of robust feature extractors from a source network, and fine-tuning end-to-end in the target domain with lifelong learning strategies
RESULT: new models inherit the robustness of their parent networks while achieving high accuracy with little data

CITATION CONTEXT
Section: Introduction
Co-cited in the same sentence: yes
Before: Robust fine-tuning (RFT) that contains an adversarial objective to learn features of adversarial data can gain adversarial robustness in downstream tasks.
Citing sentence: To further improve generalization, vanilla RFT (formulated in Eq. REF, shown in the left panel of Figure REF) optimizes both adversarial and natural objectives to learn the features of adversarial and natural data simultaneously via the FE [Citation].
After: Recently, TWINS (formulated in Eq. REF) further enhances performance in downstream tasks by incorporating vanilla RFT with a dual batch normalization (BN) module.

OUTPUT:
{{"FUNCTION": "MOTIVATION", "M_ASPECT": "LIMITATION", "POLARITY": "", "EVIDENCE": "optimizes both adversarial and natural objectives", "R_MATCH": "yes", "CONFIDENCE": "medium"}}

Now classify this citation:

MAIN PAPER (M): {m_title}
{m_narrative}

CITED PAPER (R): {r_title}
{r_narrative}

CITATION CONTEXT
Section: {section}
Co-cited in the same sentence: {co_cited}
Before: {before}
Citing sentence: {center}
After: {after}
"""

# ============================================================================
# LAYER 2: CITATION RELATION FUNCTION (v2)
# ============================================================================
# Changes vs v1: + FUTURE function; stricter EXTENDS; co-citation consistency
# rule; explicit-cue rule for CONFIDENCE "high"; hard 12-word EVIDENCE limit;
# + EXAMPLE 7 (EXTENDS) and EXAMPLE 8 (FUTURE). Inputs/outputs unchanged.
# System prompt, narrative block, M_ASPECT/POLARITY/R_MATCH/CONFIDENCE sets are
# the same as v1.

LAYER_2_RELATION_FUNCTIONS_v2 = (
    "BACKGROUND", "MOTIVATION", "USES", "EXTENDS", "SUPPORTS", "COMPARISON", "FUTURE",
)
LAYER_2_RELATION_SYSTEM_v2 = LAYER_2_RELATION_SYSTEM_v1

LAYER_2_RELATION_USER_TEMPLATE_v2 = """Classify why the MAIN PAPER (M) cites the CITED PAPER (R) in the citation context below.

FUNCTION — the role R plays for M in this context (choose exactly one):
1. BACKGROUND — R is general context for the field or problem; M does not depend on R.
2. MOTIVATION — a gap, limitation, open problem, or finding in R motivates M's work.
   Also MOTIVATION when M describes R's approach in order to point out the problem M addresses (often followed by "however", "issue", "suffers from").
3. USES — M adopts something from R, as it is or adapted into a component of M's own approach: method, model, algorithm, dataset, benchmark, evaluation protocol, software, tool, measured value, or setting (cues: "we use", "we employ", "following", "based on", "inspired by", "we relied on these data").
4. EXTENDS — R's contribution itself is the object M improves, generalizes, or updates, so that M's contribution is a new version of R's; includes M's own earlier work (cues: "we extend", "we update our previous", "we generalize", "in contrast to our previous work, we consider here"). If M's result contains R's as a special case or limit ("R's solutions are a special case of ours", "reduces to R in the limit", "generalization of R"), it IS EXTENDS. Mere agreement with R, or a defence of R, is NOT EXTENDS: use COMPARISON or SUPPORTS.
5. SUPPORTS — R's result or argument is evidence for a claim, assumption, or design choice of M that M does not test itself, or removes a concern about M's approach (cues: "has been shown to", "as demonstrated in", "has been validated").
6. COMPARISON — M sets its own approach, data, or results against R, including using R as a baseline (cues: "baseline", "consistent with", "similar to", "unlike", "complementary", "outperforms"). Set POLARITY:
   - SIMILAR: agrees, is consistent with, identical or analogous to R
   - DIFFERENT: differs from, is complementary to, outperforms, or disagrees with R
7. FUTURE — R is mentioned in a statement about work M leaves for the future: possible extensions, open questions, or links to be explored later (cues: "it would be interesting to", "it should be nice to", "future work", "we leave for future"). Use FUTURE, not MOTIVATION, when the citing sentence is about work M has not done.

M_ASPECT — which part of M this citation serves (choose exactly one): TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT.
Typical pairs: BACKGROUND→TASK, MOTIVATION→LIMITATION or TASK, USES→METHOD, EXTENDS→CONTRIBUTION or METHOD, SUPPORTS→METHOD or LIMITATION, COMPARISON→RESULT or CONTRIBUTION, FUTURE→CONTRIBUTION or TASK. Deviate when the context clearly says otherwise.

Rules:
- Decide FUNCTION from the citation context: the citing sentence first, then the sentences before and after. Use the narratives of M and R only to choose M_ASPECT and to resolve ambiguity.
- [Citation] marks where R is cited. When it stands right after a full stop it belongs to the preceding sentence; when it starts a sentence it stands for R's authors (e.g. "[Citation] proposed ...").
- Other citations in the passage have been removed, so some sentences may lack a subject (e.g. "has shown that ...") or contain empty lists. Ignore these gaps.
- Mathematical notation, equations, and "REF" placeholders are extraction residue; read around them.
- If R is co-cited with other references in the same sentence, label the role of R only. If the sentence treats the co-cited works in the same way (e.g. "in [A] and [B]"), give R the FUNCTION it would get if cited alone; references that play the same role in a sentence get the same FUNCTION. Use R's narrative only to decide what R contributes to that sentence.
- If several functions fit, choose the most specific one, in this order: EXTENDS > USES > SUPPORTS > COMPARISON > FUTURE > MOTIVATION > BACKGROUND.
- POLARITY is "" for every FUNCTION except COMPARISON.
- EVIDENCE is the shortest phrase copied verbatim from the context that signals the FUNCTION: at most 12 words, never longer.
- R_MATCH is "no" if what the context says about the cited work is clearly unrelated to R's narrative (a wrong reference), otherwise "yes". If R_MATCH is "no", still fill the other fields from the context alone.
- CONFIDENCE is "low" if the context is uninformative (e.g. mostly an equation or a heading), the function is a guess, or R_MATCH is "no"; "high" if the citing sentence or its neighbours contain an explicit cue for the chosen FUNCTION (such as the cue phrases above); "medium" only if two functions fit about equally.

Return ONLY valid JSON with no prose, no markdown, no code fences:
{{"FUNCTION": "<function>", "M_ASPECT": "<aspect>", "POLARITY": "<SIMILAR|DIFFERENT|empty>", "EVIDENCE": "<phrase from context>", "R_MATCH": "<yes|no>", "CONFIDENCE": "<high|medium|low>"}}

## EXAMPLE 1 (USES)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): Adam: A Method for Stochastic Optimization
TASK: first-order gradient-based optimization of stochastic objective functions
LIMITATION: (not stated)
CONTRIBUTION: Adam: an algorithm for first-order gradient-based optimization of stochastic objective functions based on adaptive estimates of lower-order moments
METHOD: adaptive estimates of lower-order moments, invariant to diagonal rescaling of the gradients
RESULT: computationally efficient, little memory, well suited to noisy and/or sparse gradients

CITATION CONTEXT
Section: Model and Training Details
Co-cited in the same sentence: no
Before:
Citing sentence: All our models are trained using Adam [Citation] for 5,000 epochs, where each epoch is over a chunk of 10,000 training samples.
After: Since we are generating the data, we vary the training samples from 1k to 1M, and use a validation set of 10k samples.

OUTPUT:
{{"FUNCTION": "USES", "M_ASPECT": "METHOD", "POLARITY": "", "EVIDENCE": "All our models are trained using Adam", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 2 (COMPARISON, DIFFERENT)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning
TASK: human-in-the-loop grounded language learning for artificial agents
LIMITATION: current learning methods are poorly data-efficient, making interactive human training of agents impractical
CONTRIBUTION: BabyAI: an extensible research platform for investigating human-in-the-loop grounded language learning
METHOD: an extensible suite of 19 levels of increasing difficulty and a heuristic expert agent simulating a human teacher
RESULT: current deep learning methods are not yet sample-efficient for learning a language with compositional properties

CITATION CONTEXT
Section: Related Work
Co-cited in the same sentence: no
Before: Since proposed, all bAbI tasks have been solved by novel memory architectures.
Citing sentence: An gridworld environment for embodied agents with language instructions for tasks is described in [Citation]; our work here is complementary, giving question-answer pairs based on abstracted environment histories.
After: CLEVR is a popular synthetic data for testing visual reasoning given text queries.

OUTPUT:
{{"FUNCTION": "COMPARISON", "M_ASPECT": "CONTRIBUTION", "POLARITY": "DIFFERENT", "EVIDENCE": "our work here is complementary", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 3 (BACKGROUND, co-cited)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): Attention Is All You Need
TASK: sequence transduction models for machine translation
LIMITATION: the best performing models use complex recurrent or convolutional networks, which are computationally expensive and hard to parallelize
CONTRIBUTION: the Transformer: a new simple network architecture based solely on attention mechanisms
METHOD: attention-based encoder-decoder architecture trained on WMT 2014 translation tasks
RESULT: 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on English-to-French

CITATION CONTEXT
Section: Introduction
Co-cited in the same sentence: yes
Before:
Citing sentence: Advances in machine learning (ML) architectures [Citation], large datasets and model scaling for pre-training, modeling approaches, and multiple dedicated reasoning datasets have driven progress in both building models that can succeed in aspects of “reasoning” and in automatically evaluating such capabilities.
After: This has been evident particularly in the text setting, but also in computer vision.

OUTPUT:
{{"FUNCTION": "BACKGROUND", "M_ASPECT": "TASK", "POLARITY": "", "EVIDENCE": "Advances in machine learning (ML) architectures", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 4 (SUPPORTS)

MAIN PAPER (M): A Data Source for Reasoning Embodied Agents
TASK: machine reasoning with neural models, particularly using embodied agents and database-encoded world states
LIMITATION: baseline models struggle with answering certain questions about the world-state
CONTRIBUTION: a new data generator for machine reasoning that integrates with an embodied agent, generating templated text queries and answers matched with world-states encoded into a database
METHOD: templated text queries and answers matched with world-states encoded into a database; evaluated on fine-tuned pre-trained language models and graph-structured Transformers
RESULT: baseline models can answer some questions about the world-state but struggle with others

CITED PAPER (R): TAPEX: Table Pre-training via Learning a Neural SQL Executor
TASK: table pre-training for natural language processing
LIMITATION: large-scale high-quality tabular data is scarce, hindering language model pre-training on structured tabular data
CONTRIBUTION: TAPEX: a table pre-training approach that learns a neural SQL executor over a synthetic corpus
METHOD: guiding a language model to mimic a SQL executor on synthesized executable SQL queries and their outputs
RESULT: new state-of-the-art results on WikiSQL, WikiTableQuestions, SQA and TabFact

CITATION CONTEXT
Section: Environment
Co-cited in the same sentence: no
Before: Multiple time snapshots of the context are represented by appending each successive event in the sequence.
Citing sentence: While this is in some sense simplistic, it allows ready application of large pre-trained language models for ingesting the context; and this kind of approach has been shown to be effective in [Citation].
After: Alternatively, we can leverage the relational properties of the context by representing it as a graph where objects and properties are nodes, and the connections between them are the edges.

OUTPUT:
{{"FUNCTION": "SUPPORTS", "M_ASPECT": "METHOD", "POLARITY": "", "EVIDENCE": "this kind of approach has been shown to be effective", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 5 (COMPARISON, SIMILAR — physics)

MAIN PAPER (M): Columnar magnetic structure coupled with orthorhombic distortion in the antiferromagnetic iron arsenide SrFe2As2
TASK: determining the magnetic structure and its relationship with the crystallographic structure in SrFe2As2
LIMITATION: (not stated)
CONTRIBUTION: a determination of the magnetic propagation vector, interlayer coupling, and magnetic moment of SrFe2As2
METHOD: neutron powder diffraction on polycrystalline SrFe2As2 below T0 = 205 K, with a detailed Rietveld analysis
RESULT: propagation vector q = (1 0 1); antiferromagnetic coupling along the a axis within the Fe-As layer and between layers; moment 1.01(3) μB parallel to the a-axis

CITED PAPER (R): Magnetic order in BaFe2As2, the parent compound of the FeAs based superconductors in a new structural family
TASK: magnetic order in AFe2As2 superconductor parent compounds
LIMITATION: (not stated)
CONTRIBUTION: neutron diffraction studies of BaFe2As2 demonstrating an antiferromagnetic transition concurrent with a first-order structural transition
METHOD: neutron diffraction studies of BaFe2As2
RESULT: an antiferromagnetic transition concurs with a first-order structural transition

CITATION CONTEXT
Section: (none)
Co-cited in the same sentence: yes
Before: The determined magnetic structure of SrFe_2As_{{2}} is shown in Fig. REF; the AFM coupling occurs along the longer a direction and the moment orients in the same direction.
Citing sentence: The magnetic ordering in SrFe_2As_{{2}} within the FeAs layer seems identical to that in BaFe_2As_{{2}} [Citation], in LaFeAs(O,F) and in CeFeAs(O,F), although the stacking order is not common.
After: This supports the importance of the FeAs intralayer coupling as well as the weak interlayer coupling.

OUTPUT:
{{"FUNCTION": "COMPARISON", "M_ASPECT": "RESULT", "POLARITY": "SIMILAR", "EVIDENCE": "seems identical to that in BaFe_2As_{{2}}", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 6 (MOTIVATION — narrative decides M_ASPECT)

MAIN PAPER (M): AutoLoRa: A Parameter-Free Automated Robust Fine-Tuning Framework
TASK: obtaining adversarial robustness through Robust Fine-Tuning (RFT) in deep neural networks
LIMITATION: existing RFT optimizes both adversarial and natural objectives through the feature extractor (FE), yielding divergent gradient directions that introduce instability and sensitivity to hyperparameters
CONTRIBUTION: AutoLoRa: a low-rank (LoRa) branch that disentangles RFT into two distinct components and automates the scheduling of learning rates and loss term scalars
METHOD: a low-rank branch disentangling RFT into optimizing natural objectives via the LoRa branch and adversarial objectives via the FE, with heuristic scheduling of learning rates and loss term scalars
RESULT: new state-of-the-art results across a range of downstream tasks without searching for hyperparameters

CITED PAPER (R): Adversarially robust transfer learning
TASK: robust transfer learning for producing adversarially robust neural network classifiers with scarce data
LIMITATION: data scarcity and computational limitations make it hard to produce both accurate and robust models, and adversarial training is costly
CONTRIBUTION: a method for robust transfer learning that transfers robustness from a source model to a target domain
METHOD: training classifiers on top of robust feature extractors from a source network, and fine-tuning end-to-end in the target domain with lifelong learning strategies
RESULT: new models inherit the robustness of their parent networks while achieving high accuracy with little data

CITATION CONTEXT
Section: Introduction
Co-cited in the same sentence: yes
Before: Robust fine-tuning (RFT) that contains an adversarial objective to learn features of adversarial data can gain adversarial robustness in downstream tasks.
Citing sentence: To further improve generalization, vanilla RFT (formulated in Eq. REF, shown in the left panel of Figure REF) optimizes both adversarial and natural objectives to learn the features of adversarial and natural data simultaneously via the FE [Citation].
After: Recently, TWINS (formulated in Eq. REF) further enhances performance in downstream tasks by incorporating vanilla RFT with a dual batch normalization (BN) module.

OUTPUT:
{{"FUNCTION": "MOTIVATION", "M_ASPECT": "LIMITATION", "POLARITY": "", "EVIDENCE": "optimizes both adversarial and natural objectives", "R_MATCH": "yes", "CONFIDENCE": "medium"}}

## EXAMPLE 7 (EXTENDS — M's own earlier work)

MAIN PAPER (M): Simulations of laser-driven strong-field QED with Ptarmigan: Resolving wavelength-scale interference and $\\gamma$-ray polarization
TASK: modelling nonlinear Compton scattering, Breit-Wheeler pair creation, and trident pair creation in the transition between the perturbative and nonperturbative regimes of strong-field QED
LIMITATION: accurate modelling in the transition regime, where the normalized laser amplitude a_0 is comparable to unity and wavelength-scale interference is significant, is challenging
CONTRIBUTION: a benchmarked simulation of nonlinear Compton scattering, Breit-Wheeler pair creation, and trident pair creation in the transition regime using Ptarmigan, extending previous results to linearly polarized lasers and arbitrarily polarized gamma rays
METHOD: Monte Carlo particle-tracking code Ptarmigan in the locally monochromatic approximation (LMA), benchmarked against full QED calculations for pulsed plane waves
RESULT: simulation results are accurate to a few percent across the full range of particle energies and laser intensities

CITED PAPER (R): Higher fidelity simulations of nonlinear Breit-Wheeler pair creation in intense laser pulses
TASK: nonlinear Breit-Wheeler pair production by photon-laser collisions
LIMITATION: (not stated)
CONTRIBUTION: a simulation framework for nonlinear Breit-Wheeler pair production via the locally monochromatic approximation (LMA)
METHOD: a ponderomotive simulation framework including strong-field QED effects via the LMA, compared with exact analytical results
RESULT: accurate calculation of harmonic structure, the transition from multi-photon to all-order pair creation, and total pair yield

CITATION CONTEXT
Section: Summary
Co-cited in the same sentence: yes
Before: Nevertheless, this goes considerably further than any previous code and ensures good overlap with the region in which the LCFA is accurate.
Citing sentence: In contrast to our previous work [Citation], where we examined circularly polarized lasers and unpolarized \\gamma rays, we have considered here the physically richer problem of linearly polarized lasers.
After: The broken symmetry of this case makes the numerical implementation more challenging, because the loss of azimuthal symmetry means that an additional integral must be performed to evaluate the probability rates.

OUTPUT:
{{"FUNCTION": "EXTENDS", "M_ASPECT": "CONTRIBUTION", "POLARITY": "", "EVIDENCE": "In contrast to our previous work", "R_MATCH": "yes", "CONFIDENCE": "high"}}

## EXAMPLE 8 (FUTURE)

MAIN PAPER (M): Comments on the String dual to N=1 SQCD
TASK: string duals to N=1 SQCD deformed by a quartic superpotential in the quark superfields
LIMITATION: previous results in the literature were fragmented and lacked a unified view, and some puzzles remained unresolved
CONTRIBUTION: a unified view of the string duals to N=1 SQCD with quartic superpotential deformations, including new exact and asymptotic solutions
METHOD: analysis of the string duals, extending previous results and presenting new exact and asymptotic solutions
RESULT: new exact and asymptotic solutions; a resolution to an old puzzle about the beta function; a sufficient criterion for screening

CITED PAPER (R): Heavy quark potential with dynamical flavors: a first order transition
TASK: static potential between external quark-antiquark pairs in strongly coupled gauge theories with massive dynamical flavors
LIMITATION: (not stated)
CONTRIBUTION: an analysis of the static potential using a dual string description, including a first-order phase transition between linear and Coulomb-like regimes
METHOD: dual string description of the static potential, with dependence studied on theory parameters
RESULT: a first-order phase transition between a linear and a Coulomb-like regime below a critical constituent quark mass

CITATION CONTEXT
Section: Screening as string breaking
Co-cited in the same sentence: no
Before: … Therefore, when the tip of the object comes close to the IR wall r_0\\rightarrow 0, the length diverges.
Citing sentence: It should be interesting to extend the analysis of this Section to the case in which we have massive quarks; because it could provide more hints on the first order phase transition-in terms of the mass- that occurs for the connected part of the Wilson loop, see [Citation].
After: Also, it may be interesting to use the results of this Section to improve models of AdS/QCD to incorporate screening effects.

OUTPUT:
{{"FUNCTION": "FUTURE", "M_ASPECT": "CONTRIBUTION", "POLARITY": "", "EVIDENCE": "It should be interesting to extend the analysis of this Section", "R_MATCH": "yes", "CONFIDENCE": "high"}}

Now classify this citation:

MAIN PAPER (M): {m_title}
{m_narrative}

CITED PAPER (R): {r_title}
{r_narrative}

CITATION CONTEXT
Section: {section}
Co-cited in the same sentence: {co_cited}
Before: {before}
Citing sentence: {center}
After: {after}
"""

# ============================================================================
# LAYER 2 VERSION REGISTRY -- mirrors NARRATIVE_VERSIONS: single source of
# truth for which system/template/narrative_block/label-sets go together, so
# training/train_data_generator/synthetic_context_classification_dataset_building.py
# resolves everything via --version instead of hardcoding a v* constant.
# ============================================================================
LAYER_2_VERSIONS = {
    "v1": {
        "system": LAYER_2_RELATION_SYSTEM_v1,
        "template": LAYER_2_RELATION_USER_TEMPLATE_v1,
        "narrative_block": LAYER_2_NARRATIVE_BLOCK_v1,
        "functions": LAYER_2_RELATION_FUNCTIONS_v1,
        "m_aspects": LAYER_2_M_ASPECTS_v1,
        "polarities": LAYER_2_POLARITIES_v1,
        "r_match": LAYER_2_R_MATCH_v1,
        "confidence": LAYER_2_CONFIDENCE_v1,
        "description": "Layer-2 citation relation function: FUNCTION/M_ASPECT/POLARITY/EVIDENCE/R_MATCH/CONFIDENCE",
    },
    "v2": {
        "system": LAYER_2_RELATION_SYSTEM_v2,
        "template": LAYER_2_RELATION_USER_TEMPLATE_v2,
        "narrative_block": LAYER_2_NARRATIVE_BLOCK_v1,
        "functions": LAYER_2_RELATION_FUNCTIONS_v2,
        "m_aspects": LAYER_2_M_ASPECTS_v1,
        "polarities": LAYER_2_POLARITIES_v1,
        "r_match": LAYER_2_R_MATCH_v1,
        "confidence": LAYER_2_CONFIDENCE_v1,
        "description": "Layer-2 citation relation function v2: + FUTURE function, stricter EXTENDS, "
                        "co-citation consistency rule, explicit-cue CONFIDENCE rule, hard 12-word EVIDENCE limit",
    },
}

# ============================================================
# CITATION CLASSIFIER SLM (Qwen 7B fine-tuning + inference) -- v1
# ============================================================
# Lean prompt for the fine-tuned model. Use the SAME constants to build the SFT
# file and at inference. Targets come from DeepSeek outputs generated with
# LAYER_2_RELATION_USER_TEMPLATE_v2, with CONFIDENCE removed.
# Input placeholders are identical to the DeepSeek prompt:
#   {m_title} {m_narrative} {r_title} {r_narrative} {section} {co_cited} {before} {center} {after}
# Narratives are rendered with CITATION_CLASSIFIER_SLM_NARRATIVE_BLOCK_v1
# (empty fields -> "(not stated)").

CITATION_CLASSIFIER_SLM_FUNCTIONS_v1 = (
    "BACKGROUND", "MOTIVATION", "USES", "EXTENDS", "SUPPORTS", "COMPARISON", "FUTURE",
)
CITATION_CLASSIFIER_SLM_M_ASPECTS_v1 = ("TASK", "LIMITATION", "CONTRIBUTION", "METHOD", "RESULT")
CITATION_CLASSIFIER_SLM_POLARITIES_v1 = ("SIMILAR", "DIFFERENT", "")
CITATION_CLASSIFIER_SLM_R_MATCH_v1 = ("yes", "no")
CITATION_CLASSIFIER_SLM_OUTPUT_KEYS_v1 = ("FUNCTION", "M_ASPECT", "POLARITY", "EVIDENCE", "R_MATCH")

CITATION_CLASSIFIER_SLM_NARRATIVE_BLOCK_v1 = """TASK: {TASK}
LIMITATION: {LIMITATION}
CONTRIBUTION: {CONTRIBUTION}
METHOD: {METHOD}
RESULT: {RESULT}"""

CITATION_CLASSIFIER_SLM_SYSTEM_v1 = (
    "You classify the role a cited paper R plays for a main paper M in one citation context. Output JSON only."
)

CITATION_CLASSIFIER_SLM_USER_TEMPLATE_v1 = """Classify why M cites R in the citation context. [Citation] marks R.

FUNCTION (one):
- BACKGROUND: general context; M does not depend on R.
- MOTIVATION: a gap, limitation or problem in R motivates M.
- USES: M adopts a method, model, dataset, tool, value or setting from R.
- EXTENDS: M improves, generalizes or updates R's contribution; R is a special case or limit of M's result.
- SUPPORTS: R's result is evidence for a claim or choice of M.
- COMPARISON: M sets its approach or results against R, incl. R as a baseline.
- FUTURE: R appears in work M leaves for the future.
If several fit: EXTENDS > USES > SUPPORTS > COMPARISON > FUTURE > MOTIVATION > BACKGROUND.

M_ASPECT (one): TASK, LIMITATION, CONTRIBUTION, METHOD, RESULT.
POLARITY: SIMILAR or DIFFERENT for COMPARISON, otherwise "".
EVIDENCE: verbatim phrase from the context, at most 12 words.
R_MATCH: "no" if the context clearly does not describe R, otherwise "yes".

MAIN PAPER (M): {m_title}
{m_narrative}

CITED PAPER (R): {r_title}
{r_narrative}

CITATION CONTEXT
Section: {section}
Co-cited in the same sentence: {co_cited}
Before: {before}
Citing sentence: {center}
After: {after}

Output: {{"FUNCTION": "", "M_ASPECT": "", "POLARITY": "", "EVIDENCE": "", "R_MATCH": ""}}"""

# ============================================================================
# CITATION CLASSIFIER SLM VERSION REGISTRY -- mirrors LAYER_2_VERSIONS: single
# source of truth so the SFT-build step and training/train_qlora_context_classifier.py
# (and, later, an inference script) all resolve the SLM's own prompt via
# --version instead of hardcoding a v* constant.
# ============================================================================
CITATION_CLASSIFIER_SLM_VERSIONS = {
    "v1": {
        "system": CITATION_CLASSIFIER_SLM_SYSTEM_v1,
        "template": CITATION_CLASSIFIER_SLM_USER_TEMPLATE_v1,
        "narrative_block": CITATION_CLASSIFIER_SLM_NARRATIVE_BLOCK_v1,
        "functions": CITATION_CLASSIFIER_SLM_FUNCTIONS_v1,
        "m_aspects": CITATION_CLASSIFIER_SLM_M_ASPECTS_v1,
        "polarities": CITATION_CLASSIFIER_SLM_POLARITIES_v1,
        "r_match": CITATION_CLASSIFIER_SLM_R_MATCH_v1,
        "output_keys": CITATION_CLASSIFIER_SLM_OUTPUT_KEYS_v1,
        "description": "Lean SFT/inference prompt for the fine-tuned Qwen citation classifier "
                        "(targets built from LAYER_2_RELATION_USER_TEMPLATE_v2 outputs, CONFIDENCE removed)",
    },
}
