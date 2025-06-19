# The 5th APTOS Big Data Competition SDU_VSISLAB team code submission
This repository contains the official code from the **SDU_VSISLAB** team, which secured **Second Place** in the **5th APTOS Big Data Competition**.
---

## Installation
- **Recommended Environment:**
  - Python 3.8.20
  - CUDA 11.6
  - PyTorch 1.12.0

- **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

---

## APTOS_val2 Results
To obtain the final results of **APTOS_val2**, follow these steps:

1. Run the HMM combination script:
   ```bash
   python combin_max_GaussianHMM.py
   ```

2. Save the predictions:
   ```bash
   python save_pre.py
   ```

---

## Training and Evaluation
We provide a list of scripts to replicate our results.  
Please refer to the following file for the complete process:
```text
./run_all_exps.sh
```

### Prediction on APTOS_val

Use the following command:
```bash
python3 save_predictions_offline.py phase \
  --split training \
  --backbone [TRIAL_BACKBONE] \
  --seq_len 256 \
  --resume ../output/checkpoints/phase/[TRIAL_NAME]/models/checkpoint_best_acc.pth.tar \
  --cfg configs/[TRIAL_CONFIG].yaml
```

### Prediction on APTOS_val2

Use the following command:
```bash
python3 save_predictions_offline.py phase \
  --split test \
  --backbone [TRIAL_BACKBONE] \
  --seq_len 256 \
  --resume ../output/checkpoints/phase/[TRIAL_NAME]/models/checkpoint_best_acc.pth.tar \
  --cfg configs/[TRIAL_CONFIG].yaml
```

---

## Pretrained Models

All trained models are available in:
```text
./output/checkpoints
```

---
