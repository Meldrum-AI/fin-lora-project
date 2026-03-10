# fin-lora-project


Financial Risk Control LLM Fine-tuning | CPU-Only Deployment


This project implements scenario-specific SFT (Supervised Fine-Tuning) and LoRA (Low-Rank Adaptation) for financial risk control based on Microsoft's lightweight Phi-1.5 large language model. It runs entirely on CPU with no GPU dependency, covering four core financial quantitative tasks: European option pricing, VaR (Value at Risk) calculation, default probability prediction, and credit scoring. It delivers a zero-based reproducible and low-cost solution for practical financial LLM fine-tuning and deployment.


---
Key Features
Ultra Lightweight: Only 0.06% of total parameters are fine-tuned, with the LoRA adapter size under 10MB; full training and inference pipelines can run smoothly on an ordinary PC.

CPU-Only Deployment: No dedicated GPU or computing cluster required; optimized memory layout and inference logic adapt to regular office computer environments.

Financial Scenario-Oriented: Aligns with actual risk control business logic, integrating four high-frequency financial quantitative tasks with standardized instruction templates for strong practical applicability.

Developer-Friendly Engineering: Compatible with lower-version dependencies and features zero environment compatibility issues for one-click reproduction by beginners.

Tech Stack
Python, PyTorch, HuggingFace Transformers & PEFT, LoRA, SFT, Phi-1.5, CPU Inference

