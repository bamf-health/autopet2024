from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# Example function to generate random text using GPT-2 model
def generate_random_text_gpt2(prompt, max_length=50):
    inputs = tokenizer(
        prompt, return_tensors="pt", max_length=max_length, truncation=True
    )
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Example usage
prompt = "A PET/CT image patch showing the presence of the liver, lungs, spleen, heart, and a lesion uptake in the liver."
generated_text = generate_random_text_gpt2(prompt, 100)
print("Generated Text:", generated_text)
