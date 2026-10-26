# Marine sound datasets (links and what they actually contain)

Honest catalogue. **None of these WAVs ship in this git repo.** Adapters exist for DeepShip, ShipsEar, Watkins (optional root), and ToadFishFinder (`FishCallAdapter`). The live demo uses the fish checkpoint when `artifacts/models/fish_head.joblib` exists, and otherwise a ship logistic head. `--model oracle` is the procedural twin picture.

We do **not** claim species-ID SOTA on Watkins or DCLDE.

---

## In this project (adapters, not files)

| Dataset | What it is | Labels we use | How to get it | Link |
|---------|------------|---------------|---------------|------|
| Twin-synthetic `pam_field` | 1 s procedural whale / dolphin / sperm whale / seal / fish + storm | those class names | `uaere twin --scenario pam_field` | generated |
| Twin-synthetic `busy_strait` | 1 s procedural ships | tug, cargo, tanker, passenger | `uaere twin --scenario busy_strait` | generated |
| DeepShip | 47 h, 265 ships, 4 vessel classes, 32 kHz | cargo, passenger, tanker, tug | email authors (Irfan et al. 2021) | https://doi.org/10.1016/j.eswa.2021.115270 |
| ShipsEar | Vigo ships + some natural sounds | ships + natural (mapped) | public release | https://atlanttic.uvigo.es/underwaternoise/ · https://doi.org/10.1016/j.apacoust.2016.03.008 |
| Watkins Best-of | WHOI cuts, many odontocete/mysticete species | **all mapped to `biological`** | academic WHOI download | https://cis.whoi.edu/science/B/whalesounds/ |
| ToadFishFinder v4 | Pamlico Sound oyster toadfish boatwhistles and a real negative class, 24 kHz, 1.35 s, µPa | `opsanus_tau` vs `other` | Dryad CC0 zip `wavclips_TFv4.zip` | https://doi.org/10.5061/dryad.gtht76hr9 |

---

## Passive acoustic monitoring / marine mammals

| Dataset | Description | Link |
|---------|-------------|------|
| Watkins Marine Mammal Sound Database | WHOI; decades of species-labelled cuts (best-of + full). Personal/academic use; commercial restricted. | https://cis.whoi.edu/science/B/whalesounds/ |
| MobySound | Oregon State / NOAA-linked archive of marine mammal recordings for detection/classification contests. | https://www.mobysound.org/ |
| DCLDE (Detection, Classification, Localization, and Density Estimation) | Recurring workshop + challenge sets (e.g. 2015, 2018, 2022, 2024). Click trains, tonal calls, density estimation. | https://www.dclde.org/ · workshop pages vary by year |
| NOAA NCEI Passive Acoustic Data | US federal PAM holdings (sanctuaries, EEZ). Often long, weakly labelled. | https://www.ncei.noaa.gov/products/passive-acoustic-data |
| NOAA SanctSound | 2018–2022 sanctuary soundscape project; hydrophone deployments + some event annotations. | https://www.ncei.noaa.gov/products/passive-acoustic-data · https://sanctuaries.noaa.gov/science/monitoring/sound/ |
| NOAA Fisheries PAM | Program page pointing at regional hydrophone networks. | https://www.fisheries.noaa.gov/national/science-data/passive-acoustic-data |
| Cornell Macaulay Library (marine mammals) | Vetted natural-history recordings; not a UASN benchmark. | https://www.macaulaylibrary.org/ |
| DOSITS (Discovery of Sound in the Sea) | Education + example clips, not a labelled ML corpus. | https://dosits.org/galleries/audio-gallery/ |
| FishSounds | Fish sound literature + recordings index (choruses, drums, toadfish). | https://www.fishsounds.net/ |
| OBIS-SEAMAP acoustic | Spatial marine-mammal sightings; acoustic layers vary. | https://seamap.env.duke.edu/ |
| CHAT / Watkins catalog papers | Historical WHOI catalogues that Watkins grew from. | see Watkins site above |

---

## Ships / UATR (not animals, listed so you do not mix them)

| Dataset | Description | Link |
|---------|-------------|------|
| DeepShip | 4 vessel classes, coastal, 47 h 4 min. No biology. | https://doi.org/10.1016/j.eswa.2021.115270 |
| ShipsEar | Vessel + some natural; Vigo estuary. | https://atlanttic.uvigo.es/underwaternoise/ |
| DeepShip-related UATR papers | Use DeepShip/ShipsEar as accuracy leaderboards (95–98% in 2025–2026). | see `LIT_REVIEW_2024_2026.md` |

---

## Other / adjacent

| Dataset | Description | Link |
|---------|-------------|------|
| Ocean Networks Canada acoustic | NEPTUNE/VENUS hydrophones; access via Oceans 3.0. | https://www.oceannetworks.ca/ |
| JAMSTEC / ERI Japan | Research-cruise and observatory acoustics; application required. | https://www.jamstec.go.jp/ |
| IQOE (International Quiet Ocean Experiment) | Soundscape programme, not a single WAV dump. | https://www.iqoe.org/ |
| ESC-50 / AudioSet | **Terrestrial.** Used by iKnow-audio and TinyML PAWSN papers. Not marine. | https://github.com/karolpiczak/ESC-50 · https://research.google.com/audioset/ |

---

## How this repo uses them

```bash
uaere demo --scenario pam_field --model oracle   # default: animal twin + dummy head
uaere demo --scenario pam_field --model logistic # fitted Mel head (can still dump reject)
uaere train --dataset watkins --root /path/to/watkins
```

Watkins folders → `EventClass.BIOLOGICAL` only. Fine-grained species stay out of the paper tables.

For a dated 2024–2026 paper/patent review of *methods* (not datasets), see [`LIT_REVIEW_2024_2026.md`](LIT_REVIEW_2024_2026.md).
