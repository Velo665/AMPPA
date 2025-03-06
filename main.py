from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create chat pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

from transformers import StoppingCriteriaList, MaxLengthCriteria

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=100)])

def chat(prompt):
    input_text = f"User: {prompt}\nAI Tutor:"
    response = chatbot(
        input_text, 
        max_length=150, 
        temperature=0.7,  
        top_p=0.9,  
        top_k=50,  
        truncation=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    return response[0]['generated_text']



# Test the chatbot
print(chat("Hello! How can I improve my problem-solving skills?"))