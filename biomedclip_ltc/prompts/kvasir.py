"""KVASIR-LT (HyperKvasir labeled-images) prompt registry. Three CLASS-LEVEL schemes.

  P0 : class NAME only.
  P1 : class name + a simple modality template (endoscopy).
  P2 : fine-grained, medically-checked CLASS-LEVEL endoscopic descriptions
       (mucosal appearance, landmarks, grading findings). "Plan B".

Class names recovered & VERIFIED from the image-path folders in
numpy/kvasir/dic.npy. P2 descriptions are clinically reasonable endoscopy
descriptors — PLEASE REVIEW before the final run. No patient metadata is used
(KVASIR has none); test time is image-only.
"""
PROMPT_VERSION = "kvasir-v2"

# index == MONICA label id. (folder, full_name, [fine_grained_prompts])
_MAP = [
    ("bbps-2-3", "colon with good bowel preparation", [
        "Colon with good bowel preparation, endoscopic appearance: clean mucosa "
        "with minimal residual stool and a clearly visible vascular pattern, "
        "Boston Bowel Preparation Scale 2 to 3."]),
    ("polyps", "colorectal polyp", [
        "Colorectal polyp, endoscopic appearance: a protruding sessile or "
        "pedunculated mucosal lesion with an altered surface and pit pattern."]),
    ("cecum", "cecum", [
        "Cecum, endoscopic appearance: the proximal colonic pouch showing the "
        "appendiceal orifice and converging triradiate ileocecal folds."]),
    ("dyed-lifted-polyps", "dyed and lifted polyp", [
        "Dyed and lifted polyp, endoscopic appearance: a polyp elevated by "
        "submucosal injection and stained with blue chromoendoscopy dye before "
        "resection."]),
    ("pylorus", "pylorus", [
        "Pylorus, endoscopic appearance: the round muscular opening from the "
        "gastric antrum into the duodenum."]),
    ("dyed-resection-margins", "dyed resection margin", [
        "Dyed resection margins, endoscopic appearance: the mucosal margins of a "
        "polypectomy site stained with blue dye to assess completeness of "
        "resection."]),
    ("z-line", "esophageal z-line", [
        "Esophageal z-line, endoscopic appearance: the sharp squamocolumnar "
        "junction where pale esophageal mucosa meets salmon-colored gastric "
        "mucosa."]),
    ("retroflex-stomach", "retroflex view of the stomach", [
        "Retroflex view of the stomach, endoscopic appearance: a retroflexed "
        "view of the gastric fundus and cardia showing the endoscope shaft."]),
    ("bbps-0-1", "colon with poor bowel preparation", [
        "Colon with poor bowel preparation, endoscopic appearance: mucosa "
        "obscured by solid or semisolid stool, Boston Bowel Preparation Scale 0 "
        "to 1."]),
    ("ulcerative-colitis-grade-2", "ulcerative colitis grade 2", [
        "Ulcerative colitis grade 2, endoscopic appearance: moderate "
        "inflammation with loss of vascular pattern, marked erythema, friability "
        "and erosions."]),
    ("esophagitis-a", "esophagitis grade A", [
        "Esophagitis grade A, endoscopic appearance: one or more mucosal breaks "
        "no longer than 5 mm that do not extend between mucosal folds, Los "
        "Angeles grade A."]),
    ("retroflex-rectum", "retroflex view of the rectum", [
        "Retroflex view of the rectum, endoscopic appearance: a retroflexed view "
        "of the distal rectum and anorectal junction showing the endoscope "
        "shaft."]),
    ("esophagitis-b-d", "esophagitis grade B to D", [
        "Esophagitis grade B to D, endoscopic appearance: larger or confluent "
        "mucosal breaks extending between folds, Los Angeles grade B to D."]),
    ("ulcerative-colitis-grade-1", "ulcerative colitis grade 1", [
        "Ulcerative colitis grade 1, endoscopic appearance: mild inflammation "
        "with erythema and a decreased vascular pattern without friability."]),
]

CLASS_NAMES = [full for _, full, _ in _MAP]
FOLDERS = [folder for folder, _, _ in _MAP]

TEMPLATES = [
    "an endoscopic image of {}",
    "gastrointestinal endoscopy showing {}",
    "a GI endoscopy image of {}",
    "this is an endoscopic image showing {}",
]


def class_names(label_map=None):
    return list(CLASS_NAMES)


def build(scheme, label_map=None):
    """Return list[list[str]] of prompts per class for scheme in {P0, P1, P2}."""
    out = []
    for (_, full, fg) in _MAP:
        if scheme == "P0":
            out.append([full])
        elif scheme == "P1":
            out.append([t.format(full) for t in TEMPLATES])
        elif scheme == "P2":
            out.append(list(fg))
        else:
            raise ValueError(f"unknown scheme {scheme}")
    return out
