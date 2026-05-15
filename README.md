# CoAff-DTI

CoAff-DTI is an end-to-end deep learning framework for fine-grained drug--target interaction (DTI) prediction using pre-trained language models (PLMs) and affinity-guided cross-modal interaction learning.

The framework is designed to address the limitations of conventional PLM-based DTI models that primarily rely on coarse-grained global embeddings. CoAff-DTI introduces token-level interaction modeling and affinity-guided feature fusion to better capture local interaction patterns between drug substructures and protein residues.

---

# Framework Overview

The proposed framework mainly consists of:

- **PLM-based Representation Learning**
  - Drug representations are extracted using ChemBERTa.
  - Protein representations are extracted using ESM-2.

- **Multi-view Tokenizer**
  - Decomposes global embeddings into fine-grained pharmacophore and residue tokens.

- **Affinity-Guided Cross-Attention (AGCA)**
  - Models local cross-modal interactions between drugs and proteins.

- **Affinity-Gating Fusion (AGF)**
  - Enhances multimodal feature integration through adaptive interaction modeling.

---

# Environment Setup

## Requirements

- Python 3.9
- PyTorch 2.1.1
- CUDA 12.1

## Main Dependencies

```bash
torch==2.1.1+cu121
torch-geometric==2.3.1
torch-cluster==1.6.2
torch-scatter==2.1.2
torch-sparse==0.6.18
scikit-learn==1.6.1
