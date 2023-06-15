from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline


PROMPT_INPUT = """Write a SQL query to answer the following question based a table named products, the columns of the table are list below:
Product_ID, Account_No, Daily_Price, Fy.

Question: {question}"""

model_id = "/home/kevin/llm/chatDS/output_new"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

input_text = PROMPT_INPUT.format_map({"question": "what is the average price for each product?"})

print(input_text)

pipe = pipeline(
    "text2text-generation",
    model=model, tokenizer=tokenizer, max_length=512
)
print(pipe(input_text))



