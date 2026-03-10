# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import traceback

# ===================== 0. 基础配置 =====================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:33210"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:33210"
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache"
os.environ["PYTHONIOENCODING"] = "utf-8"


class Config:
    BASE_DIR = "D:/fin-lora-project"
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    LORA_DIR = os.path.join(BASE_DIR, "lora")
    DATA_PATH = os.path.join(BASE_DIR, "fin_data.csv")
    EPOCHS = 1
    MAX_SEQ_LEN = 64
    LR = 2e-4
    BATCH_SIZE = 1
    DEVICE = "cpu"
    SFT_TEMPLATE = "<s>[INST] {instruction} [/INST] {response} </s>"


# ===================== 1. 工具函数 =====================
def init_dirs():
    for dir_path in [Config.BASE_DIR, Config.OUTPUT_DIR, Config.LORA_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"✅ D盘文件夹已创建：{Config.BASE_DIR}")


def log_info(content):
    log_path = os.path.join(Config.BASE_DIR, "train_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {content}\n")
    print(f"📌 {content}")


# ===================== 2. 核心修复：数据集直接编码为 input_ids =====================
def generate_encoded_data():
    """生成包含 input_ids/attention_mask 的数据集（适配旧版 Trainer）"""
    log_info("开始生成编码后的金融数据集（input_ids 格式）")

    # 原始SFT文本
    sft_texts = [
        "<s>[INST] 计算欧式看涨期权价格 S=100 K=95 T=1 r=0.04 sigma=0.2 [/INST] 价格=13.6932 </s>",
        "<s>[INST] 计算VaR 均值=0.0012 标准差=0.02 持仓=100万 [/INST] VaR=41500 </s>",
        "<s>[INST] 计算企业违约概率 资产负债率=0.75 流动比率=1.1 净利润率=0.02 [/INST] 违约概率=18.5% </s>",
        "<s>[INST] 计算个人信用评分 逾期次数=2 信贷利用率=0.6 信用年限=8年 [/INST] 评分=650 </s>"
    ]

    # 加载tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-1_5",
        cache_dir="D:/hf_cache",
        local_files_only=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 编码为 input_ids / attention_mask
    encoded_data = []
    for text in sft_texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=Config.MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        encoded_data.append({
            "input_ids": enc["input_ids"].flatten().numpy(),
            "attention_mask": enc["attention_mask"].flatten().numpy()
        })

    # 保存并返回Dataset
    df = pd.DataFrame(encoded_data)
    df.to_csv(Config.DATA_PATH, index=False)

    from datasets import Dataset
    dataset = Dataset.from_pandas(df)
    log_info(f"✅ 编码数据集生成完成，共{len(dataset)}条样本")
    return dataset, tokenizer


# ===================== 3. 模型初始化 =====================
def init_model(tokenizer):
    log_info("加载Phi-1.5模型并配置LoRA")

    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5",
        cache_dir="D:/hf_cache",
        local_files_only=False,
        torch_dtype=torch.float32,
        device_map=Config.DEVICE
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"✅ LoRA配置完成，可训练参数：{trainable / 1e6:.2f}M")
    return model


# ===================== 4. 修复版训练（只传 input_ids） =====================
def train_sft():
    dataset, tokenizer = generate_encoded_data()
    model = init_model(tokenizer)

    log_info("开始SFT训练（CPU模式）")
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LR,
        num_train_epochs=Config.EPOCHS,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        report_to="none",
        seed=42,
        remove_unused_columns=False
    )

    # 核心修复：DataCollatorForLanguageModeling 只传 tokenizer 和 mlm=False
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(Config.LORA_DIR)
    log_info(f"✅ 训练完成！LoRA权重保存在: {Config.LORA_DIR}")
    return model, tokenizer


# ===================== 5. 推理函数 =====================
def infer(model, tokenizer):
    log_info("开始推理金融任务")
    test_inputs = [
        "<s>[INST] 计算欧式看涨期权价格 S=110 K=105 T=1.5 r=0.05 sigma=0.3 [/INST] ",
        "<s>[INST] 计算企业违约概率 资产负债率=0.8 流动比率=0.9 净利润率=0.01 [/INST] "
    ]

    for text in test_inputs:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_SEQ_LEN
        )
        inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n📝 输入: {text[:50]}...")
        print(f"🤖 输出: {result}")
    log_info("✅ 所有任务结束！")


# ===================== 主函数 =====================
def main():
    init_dirs()
    try:
        model, tokenizer = train_sft()
        infer(model, tokenizer)
        print("\n" + "=" * 50)
        print("🎉 【终极成功】金融风控+SFT+LoRA项目完成！")
        print(f"📁 路径：{Config.BASE_DIR}")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()