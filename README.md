# SCCADE: Soliton Chaos Cellular Automata DNA-Based Image Encryption

## Overview

SCCADE (Soliton Chaos Cellular Automata DNA-Based Encryption) is a lightweight image encryption framework designed for secure multimedia transmission in IoT and fog-computing environments. The proposed scheme combines chaotic systems, DNA-based diffusion, Cellular Automata (CA), and Feedback XOR Matrix Encryption (FXME) to provide high security while maintaining computational efficiency suitable for resource-constrained devices.

The encryption process consists of:

* 128-bit chaos-based key generation
* DNA diffusion of encryption keys
* Soliton-wave-based key diffusion
* Pixel shuffling
* Cellular Automata transformation
* Feedback XOR Matrix Encryption (FXME)

The proposed method has been evaluated using statistical, differential, and randomness analyses including entropy, correlation, NPCR, UACI, SSIM, PSNR, and NIST randomness tests.

---

## Features

* Lightweight image encryption for IoT applications
* Hénon and Lorenz chaotic systems for key generation
* DNA-based key diffusion
* Soliton-wave diffusion mechanism
* Cellular Automata-based pixel transformation
* Feedback XOR Matrix Encryption (FXME)
* Image encryption and decryption support
* Statistical security analysis

---

## Repository Structure

SCCADE/
│
├── src/
│   ├── sccade.py
│   ├── analysis.py
│   
│
├── test_images/
│   ├── Jellybeans.png
│   ├── Airplane.png
│   └── Peppers.png
│   └── House.png
│
├── results/
│   ├── attacks/
│   ├── Corelation_ananlysis/
│   └── decrypted_image/
│   └── decrypted_image/
│   └── encrypted_image/
│    └── Histograms/
│    └── original_image/ 
│
├── requirements.txt
├── LICENSE
└── README.md

----

## Requirements

* Python 3.10 or higher
* NumPy
* OpenCV
* SciPy
* Matplotlib
* Scikit-image

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Encryption and Decryption

Place the input image in the project directory or upload it when prompted.

Execute:

python sccade.py on vscode version 1.123.0.

Generates:

* Encrypted image
* Decrypted Image
* Encryption keys


### Statistical Analysis

```bash
python analysis.py original_image.png encrypted_image.png
```

Calculates:

* Information Entropy
* Correlation Coefficient
* NPCR
* UACI
* MSE
* PSNR
* SSIM
* RGB Histogram Analysis

---

## Security Evaluation

The SCCADE framework has been evaluated using:

### Statistical Analysis

* Histogram Analysis
* Correlation Analysis
* Information Entropy

### Differential Analysis

* NPCR (Number of Pixel Change Rate)
* UACI (Unified Average Changing Intensity)

### Quality Analysis

* MSE
* PSNR
* SSIM

## Applications

* Internet of Things (IoT)
* Fog Computing
* Smart Healthcare
* Smart Cities
* Military Communication
* Wireless Sensor Networks
* Secure Multimedia Transmission

---



