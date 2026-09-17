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
# Narrative Extraction Prompts (3-label: TASK, LIMITATION, CONTRIBUTION)
# ============================================================================
# Version: v1 (2025-09-17)
# Purpose: Generative extraction of research narrative components from abstracts
# Labels: TASK (research area/domain), LIMITATION (gap/challenge), CONTRIBUTION (novel thing)
# Output: Structured JSON with extracted text (no spans/offsets; text only)

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