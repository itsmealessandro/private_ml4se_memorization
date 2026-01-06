from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_cuda = torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if not use_cuda:
            self.model.to("cpu")

    def generate(self, prompt, max_tokens=256, return_full_text=False):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if return_full_text:
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        prompt_len = inputs["input_ids"].shape[-1]
        completion_ids = output[0][prompt_len:]
        return self.tokenizer.decode(completion_ids, skip_special_tokens=True)
