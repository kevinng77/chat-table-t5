table_columns = "Transaction_ID, Platform, Product_ID, User_ID, Transaction_Amount, Region, Transaction_Time, Transaction_Unit, User_Comments"

table_name = "my_data"

GEN_DATA_PROMPT_PREFIX = f"""
You are asked to come up with a set of 10 diverse tasks. Each task includes a instruction and a output.
These tasks are about a table, named {table_name}. The columns of the table are list bellow:
{table_columns}

Here are the requirements:
1. Try not to repeat the verb for each question to maximize diversity.
2. The instruction should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
3. The output should be a syntactically correct SQL query, 
4. This is important! Endeavor to refer to the table columns in a varied manner within the instructions, without adhering strictly to the exact column names. For instance, 'Product_ID' could be referred to as 'product id'.
5. Do not add any symbel "\\" in the SQL Output.

List of 10 tasks:

"""

MODEL_PROMPT = """
"""
