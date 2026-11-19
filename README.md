# Opening the Black Box: Explaining Tumor and Splice-Site Classifiers

COMP 565, McGill University — Assignment 2.

Two models, three explanation methods each, and one question: do the
explanations point at biology we already know?

## Question 1 — Which features drive a breast-cancer classifier?

- Trains an **XGBoost** classifier on the Breast Cancer Wisconsin dataset
  (`sklearn.datasets.load_breast_cancer`).
- Explains it with **LIME**, **KernelSHAP** and **TreeSHAP**, each shown as a
  global top-10 bar plot, a beeswarm plot, and force plots for individual
  true/false positives and negatives.
- Compares the methods: nine of the top ten features are shared by all three,
  with *worst area* the only feature in every method's top three.

## Question 2 — What does a splice-site CNN look at?

- Trains a 1-D **convolutional network** (`SpliceCNN`, PyTorch) on one-hot DNA
  windows from the UCI splice-junction dataset (donor / acceptor / neither).
- Runs **DeepLIFT / DeepSHAP** with dinucleotide-shuffled references and plots
  per-nucleotide attributions as sequence logos.
- The donor attributions peak on the canonical **GT** dinucleotide, followed by
  a nearby **AG**, matching the extended donor consensus (GTRAGT).

## Files

- `COMP565_A2_xAI.ipynb` — full notebook for both questions, with outputs.
- `A2_xAI_CNN.py` — the splice-site CNN and DeepLIFT analysis as a script.

## Running it

`pip install xgboost lime shap torch ucimlrepo deeplift scikit-learn matplotlib`,
then run the notebook top to bottom. Both datasets download automatically.

## Author

Andrea Paola Gómez Jaime
