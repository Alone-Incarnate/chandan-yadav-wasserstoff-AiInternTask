from transformers import pipeline

# Load the text generation pipeline with the GPT-2 model
text_generator = pipeline("text-generation", model="gpt2")

# Function to generate text
def generate_text(prompt):
    generated_text = text_generator(prompt, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
    return generated_text[0]['generated_text']

# Example usage
prompt = "Once upon a time in a faraway land,"
generated_text = generate_text(prompt)
print(generated_text)
