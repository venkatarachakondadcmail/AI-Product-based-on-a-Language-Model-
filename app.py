import gradio as gr
import openai
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import io
import numpy as np
import requests

# Initialize the VQA Processor and Model
class VQAProcessor:
    def __init__(self):
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

class VQAInference:
    def __init__(self):
        self.processor = VQAProcessor()
        openai.api_key = "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"

    def process_image(self, image_or_url):
        if isinstance(image_or_url, str):
            image = Image.open(requests.get(image_or_url, stream=True).raw)
        else:
            image = Image.fromarray(np.uint8(image_or_url))
        return image

    def generate_gpt3_answer(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002", prompt=prompt, max_tokens=100
        )
        return 'Description: ' + response.choices[0].text.strip()

    def vqa_inference(self, image_or_url, question):
        image = self.process_image(image_or_url)
        inputs = self.processor.processor(text=question, images=image, return_tensors="pt")
        outputs = self.processor.vqa_model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        vqa_answer = self.processor.vqa_model.config.id2label[idx]
        prompt = f"Question: {question}\nAnswer: {vqa_answer}\n"
        gpt3_answer = prompt + "Description: "+ self.generate_gpt3_answer(prompt)
        return gpt3_answer

# Create Gradio interface
class VQAInterface:
    def __init__(self):
        self.vqa_inference = VQAInference()
        self.image_input = gr.inputs.Image(label="Upload an Image or Enter Image URL")
        self.question_input = gr.inputs.Textbox(label="Question")
        self.output_text = gr.outputs.Textbox(label="Answer")

    def launch_interface(self):
        iface = gr.Interface(
            fn=self.vqa_inference.vqa_inference,
            inputs=[self.image_input, self.question_input],
            outputs=self.output_text
        )
        iface.launch(auth=('user', 'admin'), auth_message="Enter your username and password that you received on Slack")

if __name__ == "__main__":
    vqa_interface = VQAInterface()
    vqa_interface.launch_interface()
