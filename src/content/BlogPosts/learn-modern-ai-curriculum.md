---
title: "A Practical Modern AI Curriculum: Classical ML → Deep Learning → Generative AI"
date: "2025-09-05"
excerpt: "A hands-on 12-week curriculum that moves from classical machine learning through deep learning to generative AI. Assumes familiarity with Python, linear algebra, and calculus."
tags: ["curriculum","machine-learning","deep-learning","generative-ai","education", "self study"]
---

# A Practical Modern AI Curriculum

This 12-week, project-first curriculum is designed for practitioners who already know Python, linear algebra, and calculus. It moves from classical machine learning foundations to deep learning, then to modern generative AI workflows. Each week lists key topics, short exercises, recommended libraries, and a suggested reading.

## How to use
- Follow the weekly modules sequentially.
- Use small, reproducible notebooks or scripts for exercises.
- Prefer open datasets (UCI, Kaggle, Hugging Face datasets) for practice.
- Tools: NumPy, pandas, scikit-learn, Matplotlib/Seaborn, PyTorch, Transformers, Diffusers, Hugging Face Hub.

---

## Week 1 — Practical Supervised Learning (Classical)
- Topics: problem framing, bias/variance, train/val/test, cross-validation, evaluation metrics (accuracy, precision/recall, ROC-AUC).
- Libraries: pandas, scikit-learn.
- Exercises: implement k-fold CV, compare logistic regression vs. decision trees on UCI dataset.
- Reading: scikit-learn user guide (classification).

## Week 2 — Feature Engineering & Regularization
- Topics: preprocessing, one-hot / embeddings for categorical, scaling, L1/L2 regularization, feature selection.
- Exercises: pipeline with ColumnTransformer; LASSO feature selection.
- Project milestone: build a baseline pipeline + report for a tabular problem.

## Week 3 — Ensemble Methods & Trees
- Topics: random forests, gradient boosting (XGBoost/LightGBM/CatBoost), feature importance, hyperparameter tuning.
- Exercises: grid/ Bayesian search for boosting model, interpret feature importances.
- Reading: gradient boosting guides.

## Week 4 — Probabilistic Models & Uncertainty
- Topics: Bayesian thinking, calibration, simple Bayesian linear regression, predictive intervals.
- Exercises: bootstrap vs. bayesian credible intervals; calibrate classifier probabilities.

---

## Week 5 — Introduction to Deep Learning
- Topics: computational graphs, backpropagation recap, PyTorch basics (tensors, autograd).
- Exercises: implement a small fully-connected network with PyTorch for MNIST.
- Reading: PyTorch quickstart.

## Week 6 — CNNs and Computer Vision Basics
- Topics: convolutions, pooling, transfer learning, data augmentation.
- Exercises: fine-tune ResNet on a small image dataset; try augmentation pipelines.
- Tools: torchvision, timm.

## Week 7 — Sequence Models and Attention
- Topics: RNN/LSTM overview (brief), attention mechanism, Transformer architecture.
- Exercises: implement attention mechanism; run a Transformer encoder on toy sequence tasks.
- Reading: "Attention Is All You Need" (practical recap).

## Week 8 — Deep Learning Best Practices
- Topics: optimization (Adam variants), learning rate schedules, regularization (dropout, weight decay), mixed precision, logging/experimentation.
- Exercises: train with LR finder, add early stopping, use WandB or simple logging.

Midterm project: train and deploy a small end-to-end model (classification or segmentation) with clear evaluation and reproducible training script.

---

## Week 9 — Generative Modeling Foundations
- Topics: likelihood-based models, VAEs intuition, GAN basics, autoregressive models.
- Exercises: train a small VAE on MNIST; sample and visualize reconstructions.

## Week 10 — Transformers for Generation (Text)
- Topics: causal language modeling, tokenization, decoding strategies (greedy, beam, sampling, top-k/top-p).
- Exercises: fine-tune a small Transformer LM using Hugging Face Transformers on a domain dataset.
- Tools: Transformers, Datasets, tokenizers.

## Week 11 — Diffusion & Image Generation
- Topics: diffusion model intuition, UNet architectures, classifier-free guidance, latent diffusion.
- Exercises: run a pre-trained diffusion model (e.g., Stable Diffusion / Diffusers) to generate images; experiment with guidance scale.
- Tools: diffusers, accelerate.

## Week 12 — Production, Safety, and Capstone
- Topics: model serving (FastAPI, TorchServe), inference optimization, evaluation/metrics for generative models, alignment and safety basics.
- Capstone: pick one
  - Text capstone: build a domain-specific assistant (fine-tune / retrieval-augmented generation).
  - Vision capstone: train or fine-tune a generative image pipeline and produce a reproducible demo.
- Deliverables: code repo, README, evaluation, short demo notebook.

---

## Suggested projects & exercises
- Tabular pipeline + SHAP explanations.
- Image classification with transfer learning; deploy as small web demo.
- Fine-tune a small LM and wrap with a retrieval layer.
- Create a reproducible generative image notebook (prompt → image → postprocessing).

## Recommended readings & courses
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (concepts) — skim for ML intuition.
- PyTorch docs and official tutorials.
- Hugging Face course (Transformers & Generative).
- Papers: "Attention Is All You Need", diffusion model overviews.

## Assessment & tips
- Prefer small, frequent checkpoints over single large runs.
- Version data and experiments. Use lightweight logging (WandB or plain JSON).
- Evaluate both quantitative metrics and qualitative results (especially for generative models).
- Be mindful of compute: prefer smaller models / subset datasets for rapid iteration.

---

If you want, I can:
- produce a shorter primer (one-page) or expand any week into detailed lesson notes and runnable notebooks,
- generate a starter Jupyter/Colab notebook for Week 5 (PyTorch quickstart).

To add this post: save the above exactly to `src/content/BlogPosts/learn-modern-ai-curriculum.md` and start the dev server:

```powershell
npm run dev
# or
pnpm dev
```
// filepath: src/content/BlogPosts/learn-modern-ai-curriculum.md
---
title: "A Practical Modern AI Curriculum: Classical ML → Deep Learning → Generative AI"
date: "2025-09-05"
excerpt: "A hands-on 12-week curriculum that moves from classical machine learning through deep learning to generative AI. Assumes familiarity with Python, linear algebra, and calculus."
tags: ["curriculum","machine-learning","deep-learning","generative-ai","education"]
---

# A Practical Modern AI Curriculum

This 12-week, project-first curriculum is designed for practitioners who already know Python, linear algebra, and calculus. It moves from classical machine learning foundations to deep learning, then to modern generative AI workflows. Each week lists key topics, short exercises, recommended libraries, and a suggested reading.

## How to use
- Follow the weekly modules sequentially.
- Use small, reproducible notebooks or scripts for exercises.
- Prefer open datasets (UCI, Kaggle, Hugging Face datasets) for practice.
- Tools: NumPy, pandas, scikit-learn, Matplotlib/Seaborn, PyTorch, Transformers, Diffusers, Hugging Face Hub.

---

## Week 1 — Practical Supervised Learning (Classical)
- Topics: problem framing, bias/variance, train/val/test, cross-validation, evaluation metrics (accuracy, precision/recall, ROC-AUC).
- Libraries: pandas, scikit-learn.
- Exercises: implement k-fold CV, compare logistic regression vs. decision trees on UCI dataset.
- Reading: scikit-learn user guide (classification).

## Week 2 — Feature Engineering & Regularization
- Topics: preprocessing, one-hot / embeddings for categorical, scaling, L1/L2 regularization, feature selection.
- Exercises: pipeline with ColumnTransformer; LASSO feature selection.
- Project milestone: build a baseline pipeline + report for a tabular problem.

## Week 3 — Ensemble Methods & Trees
- Topics: random forests, gradient boosting (XGBoost/LightGBM/CatBoost), feature importance, hyperparameter tuning.
- Exercises: grid/ Bayesian search for boosting model, interpret feature importances.
- Reading: gradient boosting guides.

## Week 4 — Probabilistic Models & Uncertainty
- Topics: Bayesian thinking, calibration, simple Bayesian linear regression, predictive intervals.
- Exercises: bootstrap vs. bayesian credible intervals; calibrate classifier probabilities.

---

## Week 5 — Introduction to Deep Learning
- Topics: computational graphs, backpropagation recap, PyTorch basics (tensors, autograd).
- Exercises: implement a small fully-connected network with PyTorch for MNIST.
- Reading: PyTorch quickstart.

## Week 6 — CNNs and Computer Vision Basics
- Topics: convolutions, pooling, transfer learning, data augmentation.
- Exercises: fine-tune ResNet on a small image dataset; try augmentation pipelines.
- Tools: torchvision, timm.

## Week 7 — Sequence Models and Attention
- Topics: RNN/LSTM overview (brief), attention mechanism, Transformer architecture.
- Exercises: implement attention mechanism; run a Transformer encoder on toy sequence tasks.
- Reading: "Attention Is All You Need" (practical recap).

## Week 8 — Deep Learning Best Practices
- Topics: optimization (Adam variants), learning rate schedules, regularization (dropout, weight decay), mixed precision, logging/experimentation.
- Exercises: train with LR finder, add early stopping, use WandB or simple logging.

Midterm project: train and deploy a small end-to-end model (classification or segmentation) with clear evaluation and reproducible training script.

---

## Week 9 — Generative Modeling Foundations
- Topics: likelihood-based models, VAEs intuition, GAN basics, autoregressive models.
- Exercises: train a small VAE on MNIST; sample and visualize reconstructions.

## Week 10 — Transformers for Generation (Text)
- Topics: causal language modeling, tokenization, decoding strategies (greedy, beam, sampling, top-k/top-p).
- Exercises: fine-tune a small Transformer LM using Hugging Face Transformers on a domain dataset.
- Tools: Transformers, Datasets, tokenizers.

## Week 11 — Diffusion & Image Generation
- Topics: diffusion model intuition, UNet architectures, classifier-free guidance, latent diffusion.
- Exercises: run a pre-trained diffusion model (e.g., Stable Diffusion / Diffusers) to generate images; experiment with guidance scale.
- Tools: diffusers, accelerate.

## Week 12 — Production, Safety, and Capstone
- Topics: model serving (FastAPI, TorchServe), inference optimization, evaluation/metrics for generative models, alignment and safety basics.
- Capstone: pick one
  - Text capstone: build a domain-specific assistant (fine-tune / retrieval-augmented generation).
  - Vision capstone: train or fine-tune a generative image pipeline and produce a reproducible demo.
- Deliverables: code repo, README, evaluation, short demo notebook.

---

## Suggested projects & exercises
- Tabular pipeline + SHAP explanations.
- Image classification with transfer learning; deploy as small web demo.
- Fine-tune a small LM and wrap with a retrieval layer.
- Create a reproducible generative image notebook (prompt → image → postprocessing).

## Recommended readings & courses
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" (concepts) — skim for ML intuition.
- PyTorch docs and official tutorials.
- Hugging Face course (Transformers & Generative).
- Papers: "Attention Is All You Need", diffusion model overviews.

## Assessment & tips
- Prefer small, frequent checkpoints over single large runs.
- Version data and experiments. Use lightweight logging (WandB or plain JSON).
- Evaluate both quantitative metrics and qualitative results (especially for generative models).
- Be mindful of compute: prefer smaller models / subset datasets for rapid iteration.
