from .Mapping import table_gen
from .NL2SQL import query_gen
from .PipelineExec import query_exec, display
from .util import num_tokens_from_messages, num_tokens_from_functions
import autopipeline

function_stage = {"table_gen": "Table Generation", "query_gen": "Query Generation", "query_exec": "Query Execution", "display": "Result Evaluation"}
input_var = {"table_gen": ["query", "table", "columns", "desc", "status", "function_chain", "verbose", "client", "udf", "gpt4"], "query_gen": ["query", "columns", "desc", "status", "verbose", "client", "gpt4"], "query_exec": ["table", "code", "status", "verbose"], "display": ["result", "status", "verbose"]}
res_var = {"table_gen": ["table", "columns", "desc", "status", "require_new", "feedback"], "query_gen": ["code", "status"], "query_exec": ["result", "status", "require_new", "feedback"], "display": ["status"]}

def pipeline_gen(query, table, function_chain, desc, verbose, client, udf, gpt4):
    status = []
    columns = table.columns.tolist()
    value_dict = {"query": query, "table": table, "desc": desc, "columns": columns, "status": status, "function_chain":function_chain, "verbose": verbose, "client": client, "udf": udf, "gpt4": gpt4}
    while 'displayed' not in status:
        columns = value_dict["columns"]
        desc = value_dict["desc"]
        status = value_dict["status"]
        if verbose:
            print("VERBOSE:"+"Current completed stages: ", status)
            print("VERBOSE:"+"Current table descriptions: ", desc)

        response = pipeline_gpt(query, columns, desc, status, verbose, client, gpt4)

        try:
            func = response.function_call
            f = func.name
        except:
            feedback = response.content
            return True, feedback, None, None
        
        print("********** Start", function_stage[f])
        input_vars = [value_dict[v] for v in input_var[f]]
        values = globals()[f](*input_vars)

        if "require_new" in res_var[f] and values[-2]:
            return True, values[-1], None, None
        for name, value in zip(res_var[f], values):
            value_dict[name] = value
        if len(status) >= 3 and status[-1] == status[-2] == status[-3]: # keep looping
            return True, values[-1], None, None
        print("********** Complete", function_stage[f])
    return False, "", value_dict["result"], value_dict["table"]

def pipeline_gpt(user_query, column_description, description, status, verbose, client, gpt4):
    functions = [
        {
            "name": "table_gen", 
            "description": "This function should be called when the given table does not include all the columns required to answer users' query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The base table.",
                    }
                },
                "required": ["table"]
            }
        },
        {
            "name": "query_gen",
            "description": "This function should be called when the given table already includes all the information required to answer users' query ('table augmented'), but the code for answering the queries are not yet provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The augmented table.",
                    }
                },
                "required": ["table"]
            }
        },
        {
            "name": "query_exec",
            "description": "This function should be called when the given table already includes all the information required to answer users' query ('table augmented'), the code for answering the queries are also provided ('code generated'), however, the execution results are not available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The augmented table.",
                    },
                    "code": {
                        "type": "string",
                        "description": "The execution code.",
                    },
                },
                "required": ["table", "code"]
            }
        },
        {
            "name": "display",
            "description": "This function should be called when the given table already includes all the information required to answer users' query ('table augmented'), the code for answering the queries are also provided ('code generated'), and the execution results are available ('code executed'), but the results have not been displayed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The result to be displayed.",
                    }
                },
                "required": ["table"]
            }
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "I want to count the number of positive summaries on the cases. I am given a table with the following columns: ['id', 'name', 'name_abbreviation', 'decision_date', 'case'], where 'id' is the case ID; 'name' is in the format of A v. B, where A and B are the accuser and the defendent on the court, and 'name_abbreviation' is the same as 'name', except that their names are in abbreviations (when querying a case with name, you should check both their whole name and their abbreviates); 'decision_date' is the date of the court, e.g., '1987-11-09'; 'case' is the sentence/description of the jury. The current completed tasks are []"
        },
        {
            "role": "assistant",
            "content": "To count the number of positive summaries on the cases, the user should first get the summaries on the cases, and then get the sentiment of these summaries. However, the provided table does not provide information about these two properties. The table should be augmented using the 'table_gen' function.",
            "function_call": {
                "name": "table_gen",
                "arguments": "{\n  \"table\": \"table\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive summaries on the cases. I am given a table with the following columns: 'id' is the case ID; 'name' is in the format of A v. B, where A and B are the accuser and the defendent on the court, and 'name_abbreviation' is the same as 'name', except that their names are in abbreviations (when querying a case with name, you should check both their whole name and their abbreviates); 'decision_date' is the date of the court, e.g., '1987-11-09'; 'case' is the sentence/description of the jury; 'case_summary' column provides summaries of the 'case' column; 'case_summary_sentiment' is the sentiment of the content of the 'case_summary' column being one of 'Positive', 'Negative', or 'Neutral'; The current completed tasks are ['table augmented']"
        },
        {
            "role": "assistant",
            "content": "'table augmented' is also included in the completed tasks. However, the code has not been yet generated. Thus, the code should be generated using the 'query_gen' function.",
            "function_call": {
                "name": "query_gen",
                "arguments": "{\n  \"table\": \"table\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive summaries on the cases. I am given a table with the following columns: 'id' is the case ID; 'name' is in the format of A v. B, where A and B are the accuser and the defendent on the court, and 'name_abbreviation' is the same as 'name', except that their names are in abbreviations (when querying a case with name, you should check both their whole name and their abbreviates); 'decision_date' is the date of the court, e.g., '1987-11-09'; 'case' is the sentence/description of the jury; 'case_summary' column provides summaries of the 'case' column; 'case_summary_sentiment' is the sentiment of the content of the 'case_summary' column being one of 'Positive', 'Negative', or 'Neutral'; The current completed tasks are ['table augmented', 'code generated']"
        },
        {
            "role": "assistant",
            "content": "'table augmented' and 'code generated' are already included in the completed tasks. However, the code has not been yet executed. Thus, the code should be executed using the 'query_exec' function.",
            "function_call": {
                "name": "query_exec",
                "arguments": "{\n  \"table\": \"table\"\n, \"code\": \"code\"\n}"
            }
        },
        {
            "role": "user",
            "content": "I want to count the number of positive summaries on the cases. I am given a table with the following columns: 'id' is the case ID; 'name' is in the format of A v. B, where A and B are the accuser and the defendent on the court, and 'name_abbreviation' is the same as 'name', except that their names are in abbreviations (when querying a case with name, you should check both their whole name and their abbreviates); 'decision_date' is the date of the court, e.g., '1987-11-09'; 'case' is the sentence/description of the jury; 'case_summary' column provides summaries of the 'case' column; 'case_summary_sentiment' is the sentiment of the content of the 'case_summary' column being one of 'Positive', 'Negative', or 'Neutral'; The current completed tasks are ['table augmented', 'code generated', 'code executed']"
        },
        {
            "role": "assistant",
            "content": "'table augmented', 'code generated', and 'code executed' are already included in the completed tasks. However, the result has not been yet displayed. Thus, the code should be executed using the 'display' function.",
            "function_call": {
                "name": "display",
                "arguments": "{\n  \"result\": \"result\"\n}"
            }
        },
        {
            "role": "user",
            "content": user_query  + " I am given a table with the following columns: " + str(column_description)+ " where " + description + "The current completed tasks are " + str(status)
        }
    ]

    if gpt4:
        response = client.chat.completions.create(
            model="gpt-4-0613",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        if verbose:
            num_token_msg = num_tokens_from_messages(messages, "gpt-4-0613")
            num_token_func = num_tokens_from_functions(functions, "gpt-4-0613")
            print("VERBOSE:"+"Number of tokens of messages for 'pipeline_gpt': ", num_token_msg)
            print("VERBOSE:"+"Number of tokens of functions for 'pipeline_gpt': ", num_token_func)
            print("VERBOSE:"+"Number of prompt tokens for 'pipeline_gpt': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'pipeline_gpt': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'pipeline_gpt': ", response.usage.total_tokens)
            current_pize = 0.00003 * response.usage.prompt_tokens + 0.00006 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'pipeline_gpt': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")
    
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        if verbose:
            num_token_msg = num_tokens_from_messages(messages, "gpt-3.5-turbo-0125")
            num_token_func = num_tokens_from_functions(functions, "gpt-3.5-turbo-0125")
            print("VERBOSE:"+"Number of tokens of messages for 'pipeline_gpt': ", num_token_msg)
            print("VERBOSE:"+"Number of tokens of functions for 'pipeline_gpt': ", num_token_func)
            print("VERBOSE:"+"Number of prompt tokens for 'pipeline_gpt': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'pipeline_gpt': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'pipeline_gpt': ", response.usage.total_tokens)
            current_pize = 0.0000005 * response.usage.prompt_tokens + 0.0000015 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'pipeline_gpt': ${current_pize}")
            autopipeline.cost += current_pize
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")
    
    return response.choices[0].message


