# https://gist.github.com/younesbelkada/8bb36332cd2147c070b52ab25878c78f
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training, LoraConfig, set_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
    default_data_collator,
    set_seed,
)
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer
from datasets import load_from_disk
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_utils import get_last_checkpoint

# logger = logging.getLogger(__name__)

########################################################################
#
# Utility Classes for Trainer for PEFT
#
########################################################################


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control


########################################################################
#
# Utility methods for FSDP
#
########################################################################


def safe_save_model_for_hf_trainer(trainer: Trainer, tokenizer: AutoTokenizer, output_dir: str):
    """Helper method to save model for HF Trainer."""
    # see: https://github.com/tatsu-lab/stanford_alpaca/issues/65
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    model = trainer.model
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
    if trainer.args.should_save:
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        tokenizer.save_pretrained(output_dir)


########################################################################
#
# Utility methods for peft
#
########################################################################


def create_and_prepare_model(
    model_id,
    use_4bit=False,
    use_8bit=False,
    target_modules=None,
    trust_remote_code=False,
    gradient_checkpointing=False,
    torch_dtype=torch.float32,
    fsdp=False,
    deepspeed=False,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    print("Creating and preparing model...")
    print(f"Model ID: {model_id}")
    print(f"Use 4bit: {use_4bit}")
    print(f"Use 8bit: {use_8bit}")
    print(f"Target modules: {target_modules}")
    print(f"Trust remote code: {trust_remote_code}")
    print(f"Gradient checkpointing: {gradient_checkpointing}")
    print(f"kwargs: {kwargs}")

    device_map = "auto"
    bnb_config = None
    if fsdp or deepspeed:
        device_map = None
        torch_dtype = None
        bnb_config = None

    if use_4bit:
        device_map = {"": 0}
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=False,
        )

    if use_8bit:
        raise NotImplementedError("8bit quantization is not implemented yet.")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb_config,
        use_cache=False if gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )

    # enable gradient checkpointing when quantization is disabled
    if gradient_checkpointing and use_8bit is False and use_4bit is False:
        model.gradient_checkpointing_enable()

    # create LoraConfig from kwargs
    if target_modules is not None:
        # prepare model for training
        if use_4bit or use_8bit:
            model = prepare_model_for_kbit_training(model)

        lora_alpha = getattr(kwargs, "lora_alpha", 8)
        lora_dropout = getattr(kwargs, "lora_dropout", 0.05)
        bias = getattr(kwargs, "bias", "none")

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        # initialize peft model
        model = get_peft_model(model, peft_config)
        # print parameters
        model.print_trainable_parameters()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    # set pad token to eos token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


########################################################################
# This is a fully working simple example to use peft with FSDP, Deepspeed
#
# This example fine-tunes Falcon 7B on the Dolly dataset.
#
########################################################################


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
    )
    target_modules: List[str] = field(
        metadata={"help": "List of target modules for LoRA."},
        default=None,
    )

    use_4bit: bool = field(
        metadata={"help": "Whether to use 4bit quantization."},
        default=False,
    )
    use_8bit: bool = field(
        metadata={"help": "Whether to use 8bit quantization."},
        default=False,
    )
    trust_remote_code: bool = field(
        metadata={"help": "Whether to trust remote code."},
        default=False,
    )
    # Additional LoRA Arguments
    lora_alpha: Optional[float] = field(
        metadata={"help": "Alpha value for LoRA."},
        default=None,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "Dropout value for LoRA."},
        default=None,
    )
    merge_weights: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )
    # other arguments
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Number of samples to train on"})


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    script_args = cast(ScriptArguments, script_args)
    training_args = cast(TrainingArguments, training_args)

    # script checks
    if len(training_args.fsdp) > 0 and script_args.target_modules is not None:
        raise ValueError(
            "FSDP and LoRA cannot be used together, see: https://github.com/huggingface/peft/tree/main#caveats"
        )
    if len(training_args.fsdp) > 0 and (script_args.use_4bit):
        raise ValueError("FSDP and quantization cannot be used together.")

    # set seed
    set_seed(training_args.seed)

    # get dtype
    torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32

    # create and prepare model
    model, tokenizer = create_and_prepare_model(
        **vars(script_args),
        torch_dtype=torch_dtype,
        fsdp=True if len(training_args.fsdp) > 0 else False,
        deepspeed=True if training_args.deepspeed else False,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )

    # load dataset from disk
    dataset = load_from_disk(script_args.dataset_path)
    # load dataset from disk and tokenizer
    if script_args.max_train_samples is not None:
        dataset = dataset.shuffle().select(range(script_args.max_train_samples))
        print((f"Resized dataset to {len(dataset)} samples."))

    # add callbacks
    callbacks = None
    if script_args.target_modules is not None:
        callbacks = [PeftSavingCallback()]

    # create trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    # train model
    print("Start training...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    if get_last_checkpoint(training_args.output_dir) is not None:
        # logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # save model
    if len(training_args.fsdp) > 0:
        # save model and tokenizer for easy inference
        safe_save_model_for_hf_trainer(trainer, tokenizer, training_args.output_dir)
        dist.barrier()

    # save model and tokenizer for easy inference
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
