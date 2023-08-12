from typing import Union
from transformers import Pipeline
from PIL import Image
 # You need to import the image loading function from your code
import requests
from PIL import Image
import io

def load_image(image_path_or_url, timeout=None):
    if image_path_or_url.startswith(("http://", "https://")):
        response = requests.get(image_path_or_url, stream=True, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    else:
        image = Image.open(image_path_or_url)
    
    return image
class VisualQuestionAnsweringPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom initialization if needed

    def _sanitize_parameters(self, top_k=None, padding=None, truncation=None, timeout=None, **kwargs):
        preprocess_params, postprocess_params = {}, {}
        # Custom sanitization logic
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, padding=False, truncation=False, timeout=None):
        image = load_image(inputs["image"], timeout=timeout)  # Use your image loading function
        model_inputs = self.tokenizer(
            inputs["question"], return_tensors=self.framework, padding=padding, truncation=truncation
        )
        image_features = self.image_processor(images=image, return_tensors=self.framework)
        model_inputs.update(image_features)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, top_k=5):
        if top_k > self.model.config.num_labels:
            top_k = self.model.config.num_labels

        if self.framework == "pt":
            probs = model_outputs.logits.sigmoid()[0]
            scores, ids = probs.topk(top_k)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "answer": self.model.config.id2label[_id]} for score, _id in zip(scores, ids)]

# You can then use your custom pipeline
oracle = VisualQuestionAnsweringPipeline(model="dandelin/vilt-b32-finetuned-vqa")
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
results = oracle(image=image_url, question="What is she wearing ?")
print(results)
