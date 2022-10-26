import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer



class PythonPredictor:
    def __init__(self, config):
        """This method is required. It is called once before the API 
        becomes available. It performes the setup such as downloading / 
        initializing the model.

        :param config (required): Dictionary passed from API configuration.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {device}")

        self.device = device
        self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")




    def predict(self, payload):
        """This method is required. It is called once per request. 
        Preprocesses the request payload, runs inference, and 
        postprocesses the inference output.

        :param payload (optional): The request payload
        :returns : Prediction or a batch of predictions.
        """
        # gen_tokens = self.model.generate(payload["text"], do_sample=True, temperature=0.9, max_length=100)
        # self.tokenizer.batch_decode(gen_tokens)[0]
        input_ids = self.tokenizer(payload["text"], return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
        input_ids,
        temperature=0.9,
        max_length=300,
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        return gen_text
