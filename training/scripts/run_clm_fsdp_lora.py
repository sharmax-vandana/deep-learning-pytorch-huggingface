# https://gist.github.com/younesbelkada/8bb36332cd2147c070b52ab25878c78f
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from trl import SFTTrainer

########################################################################
#
# Utility Classes for SFTTrainer for PEFT
#
########################################################################


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


# class LoadBestPeftModelCallback(TrainerCallback):
#     def on_train_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         **kwargs,
#     ):
#         print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
#         best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
#         adapters_weights = torch.load(best_model_path)
#         model = kwargs["model"]
#         set_peft_model_state_dict(model, adapters_weights)
#         return control


########################################################################
#
# Utility methods for peft
#
########################################################################


def create_and_prepare_model(
    model_id, use_4bit=False, use_8bit=False
) -> Tuple[PreTrainedModel, LoraConfig, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


########################################################################
# This is a fully working simple example to use trl's SFTTrainer.
#
# This example fine-tunes Falcon 40B on the Dolly dataset.
#
########################################################################


def format_dolly(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    model_id: Optional[str] = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_id: Optional[str] = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "The maximum sequence length for the SFTTainer to group together."}
    )


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # load model and tokenizer
    model, peft_config, tokenizer = create_and_prepare_model(script_args.model_id)
    # deactivate cache
    model.config.use_cache = (
        False if training_args.gradient_checkpointing else True,
    )  # this is needed for gradient checkpointing
    # load dataset
    dataset = load_dataset(script_args.dataset_id, split="train")

    # create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        packing=True,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        formatting_func=format_dolly,  # formatting function
        args=training_args,
        callbacks=[PeftSavingCallback()],
    )
    # train
    trainer.train()

    # save model
    dist.barrier()


if __name__ == "__main__":
    main()


# python run_clm_fsdp_lora.py \
#  --model_id tiiuae/falcon-7b \
#  --dataset_id "databricks/databricks-dolly-15k" \
#  --per_device_train_batch_size 1 \
#  --epochs 1 \
#  --optimizer adamw_apex_fused \
#  --lr 2e-4 \
#  --gradient_checkpointing True \
#  --bf16 True \
#  --tf32 True \
#  --output_dir ./tmp \
#  --logging_steps 10

# \
#  --fsdp "full_shard auto_wrap" \
#  --fsdp_transformer_layer_cls_to_wrap "DecoderLayer"
