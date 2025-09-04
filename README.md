# image-captioning-phase2

Struttura della repo:
.
├── config/
│   └── flickr8k.yaml
├── data/
│   └── Flickr8k/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── src/
│   ├── config_utils.py
│   ├── data_utils.py
│   ├── evaluation_utils.py
│   ├── formatting_utils.py
│   ├── model.py
│   ├── training_utils.py
│   ├── train.py
│   └── test.py
├── convert_flickr8k_to_framework.py
├── dataset_flickr8k_clean.json #viene fuori da una serie di operazioni di preprocessing fatte nella repo in cui ho implementato le architetture tradizionali encoder-decoder
└── README.md

Manca la cartella data/Images che contiene tutte le immagini del dataset flickr8k.
