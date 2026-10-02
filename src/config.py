from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data_2_final"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
AMOUNT_TOLERANCE = 1.0

# Exceptional aliases and compound names from registry to canonical ETM subagents.
AGENT_MAP = {
    # compound english brands
    "ak bulakvoyage": "ak bulak voyage",
    "ala tootravel": "ala too travel",
    "aylanatour": "aylana tour",
    "bishkekfly": "bishkek fly",
    "bluebirdtravel": "blue bird travel",
    "issyk kulair": "issyk kul air",
    "jailootrip": "jailoo trip",
    "karakolexpress": "karakol express",
    "kumtortours": "kumtor tours",
    "nomadtrip": "nomad trip",
    "silkrouteair": "silk route air",
    "skyway": "skyway travel",
    "sky way travel": "skyway travel",
    "skywaytravel": "skyway travel",
    "steppetravel": "steppe travel",
    "sunrisetours": "sunrise tours",
    "tengritrip": "tengri trip",
    # individuals without patronymics in registry
    "асанов кубат": "асанов кубат чынаревич",
    "асанова каныкей": "асанова каныкей эсенбекевна",
    "джолдош уулу": "джолдош уулу руслан",
    "джолдошова салтанат": "джолдошова салтанат сатаревна",
    "жумабек кызы": "жумабек кызы умут",
    "кадырова айнура": "кадырова айнура жумабекевна",
    "кенешова чолпон": "кенешова чолпон таалайовна",
    "мамбет уулу": "мамбет уулу чынгыз",
    "ниязбеков кубат": "ниязбеков кубат таалайевич",
    "ниязбекова асель": "ниязбекова асель сатаровна",
    "сатар уулу": "сатар уулу санжар",
    "сатарова жылдыз": "сатарова жылдыз жумабековна",
    "урматов эрлан": "урматов эрлан сагынович",
    "урматова айпери": "урматова айпери нурбековна",
    "чынар уулу": "чынар уулу эмир",
    "чынарова бурул": "чынарова бурул джолдошовна",
    "эсенбек уулу": "эсенбек уулу азамат",
    "эсенбеков кубат": "эсенбеков кубат элдияревич",
    "эсенбекова айгуль": "эсенбекова айгуль нурбекевна",
}

LEGAL_FORMS = ("осоо", "ип", "чп", "зао", "ооо", "ао", "llc")

