# CNN-BiLSTM Ethereum Price Prediction

Implementation of a hybrid CNN-BiLSTM deep learning architecture for Ethereum (ETH) price prediction using TensorFlow.

This repository is based on the research paper:

> **Application of CNN-BiLSTM Algorithm for Ethereum Price Prediction**  
> Published in *Journal of Applied Informatics and Computing (JAIC)*, Vol. 9 No. 4 (2025)

---

## Research Publication

### Paper Information

- **Title:** Application of CNN-BiLSTM Algorithm for Ethereum Price Prediction
- **Authors:** Hakam Dzakwan Diash, Vannesa Nathania, Mohammad Idhom, Trimono
- **Journal:** Journal of Applied Informatics and Computing (JAIC)
- **Volume:** Vol. 9 No. 4 (2025)
- **Published:** August 2025
- **DOI:** https://doi.org/10.30871/jaic.v9i4.9757
- **Paper Link:** https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/9757

### Research Overview

This research proposes a hybrid deep learning architecture combining:

- **Convolutional Neural Network (CNN)**  
  Used to extract local temporal features from Ethereum historical closing price data.

- **Bidirectional Long Short-Term Memory (BiLSTM)**  
  Used to capture bidirectional sequential dependencies and temporal patterns in time series forecasting.

The dataset consists of Ethereum daily closing prices collected from Yahoo Finance from January 2020 to January 2025.

The model was trained using:
- Early stopping mechanism
- MinMax normalization
- Sliding window sequence generation
- 100 training epochs

## Project Structure

```bash
CNN-BiLSTM_Implementation/
│
├── data/
│   ├── loader.py
│   ├── preprocessing.py
│   └── sequence.py
│
├── models/
│   └── cnn_bilstm.py
│
├── utils/
│   ├── metrics.py
│   ├── plotting.py
│   └── seed.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/hakamdzakwan/CNN-BiLSTM_Implementation.git
cd CNN-BiLSTM_Implementation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run training and evaluation:

```bash
python main.py
```

---

## Model Architecture

```text
Input
 ↓
Conv1D
 ↓
MaxPooling1D
 ↓
BiLSTM
 ↓
Dropout
 ↓
BiLSTM
 ↓
Dense
 ↓
Dense
 ↓
Output
```

---

## Citation

If you use this repository or research, please cite:

```bibtex
@article{
  title={Application of CNN-BiLSTM Algorithm for Ethereum Price Prediction},
  author={Diash, Hakam Dzakwan and Nathania, Vannesa and Idhom, Mohammad and Trimono, Trimono},
  journal={Journal of Applied Informatics and Computing},
  volume={9},
  number={4},
  pages={1709--1714},
  year={2025},
  doi={10.30871/jaic.v9i4.9757}
}
```

---

## License

This project is intended for research and educational purposes.