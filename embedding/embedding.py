import onnxruntime as rt
from transformers import AutoTokenizer
from typing import List

from embedding import config

from embedding.utils import normalize_embedding, average_pooling


class mE5Embedding:
    def __init__(self):
        self.providers = ["CPUExecutionProvider"]
        self.model = rt.InferenceSession(config.ME5_MODEL_SMALL_PATH, providers=self.providers)
        self.tokenizer = AutoTokenizer.from_pretrained(config.ME5_TOKENIZER_PATH)

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        super().__init__()

    def get_input_names(self):
        model_inputs = self.model.get_inputs()
        return [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_names(self):
        model_outputs = self.model.get_outputs()
        return [model_outputs[i].name for i in range(len(model_outputs))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            res = []
            for text in texts:
                inputs = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='np')
                outputs = self.model.run(self.output_names,
                                         {self.input_names[0]: inputs["input_ids"],
                                          self.input_names[1]: inputs["attention_mask"]})
                embeddings = normalize_embedding(average_pooling(outputs[0], inputs["attention_mask"]))
                res += embeddings.tolist()
            return res
        except Exception as e:
            print(f"Error: {e}")
            raise e
