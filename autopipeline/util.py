import tiktoken
import autopipeline
import numpy as np
import pandas as pd

# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        # print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            value = str(value)
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/11
def num_tokens_from_functions(functions, model):
        """Return the number of tokens used by a list of functions."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        num_tokens = 0
        for function in functions:
            function_tokens = len(encoding.encode(function['name']))
            function_tokens += len(encoding.encode(function['description']))
            
            if 'parameters' in function:
                parameters = function['parameters']
                if 'properties' in parameters:
                    for propertiesKey in parameters['properties']:
                        function_tokens += len(encoding.encode(propertiesKey))
                        v = parameters['properties'][propertiesKey]
                        for field in v:
                            if field == 'type':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['type']))
                            elif field == 'description':
                                function_tokens += 2
                                function_tokens += len(encoding.encode(v['description']))
                            elif field == 'enum':
                                function_tokens -= 3
                                for o in v['enum']:
                                    function_tokens += 3
                                    function_tokens += len(encoding.encode(o))
                            else:
                                print(f"Warning: not supported field {field}")
                    function_tokens += 11

            num_tokens += function_tokens

        num_tokens += 12 
        return num_tokens

def formalize_desc(d):
    res = ""
    for column_name, column_desc in d.items():
        res += "'" + column_name + "' column contains " 
        res += column_desc + "; "
        # if "column-value" in d and len(d["column-value"]) > 0:
        #     res += "IMPORTANT: the values in '"+d["column-name"]+"' column are "+d["column-desc"] + "; "
    return res

def check_alias(enum, all_description, new_description, verbose, client, gpt4):
    messages = [
        {
            "role": "system",
            "content": '''The user is going to provide you with the following:
            (1) Existing table columns;
            (2) Detailed description for each existing column;
            (3) Detailed description for the new column to be added;
            '''
            + 
            '''Your task is to check whether there already exists a column in the given table that contains the EXACTLY same information as the column to be added.
                If yes, you should reply with that specific column; if no, you should reply with an empty string.
                In either cases, you should provide the rationale.
                Your output format can ONLY be "True"/"False" + "#" + "{column name}"/"" + "#{rationale}"
                '''
        },
        {
            "role": "user",
            "content": '''Existing table columns: ['text']'''  
            + '''. The detailed description for each existing column: 'text' column contains the posts to be analyzed; '''
            + '''. The detailed description for the new column to be added: 'text_emotion' column provides emotion identified from the 'text' column. IMPORTANT: emotion values of the 'text_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'.'''
        },
        {
            "role": "assistant",
            "content": "False##The description for the new column does not match any of the existing columns.",
        },
        {
            "role": "user",
            "content": '''Existing table columns: ['text', 'text_emotion']'''  
            + '''. The detailed description for each existing column: 'text' column contains the posts to be analyzed; 'text_emotion' column provides emotion identified from the 'text' column. IMPORTANT: emotion values of the 'text_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'; '''
            + '''. The detailed description for the new column to be added: 'text_sentiment' column is the sentiment of the content of the 'text' column. IMPORTANT: sentiment values of 'text_sentiment' column can only be either 'pos', 'neg', or 'other'.'''
        },
        {
            "role": "assistant",
            "content": "False##Although 'emotion' and 'sentiment' seem to be of same meaning, they (1) are not EXACTLY the same, and (2) have different values. Thus, the description for the new column does not match any of the existing columns.",
        },
        {
            "role": "user",
            "content": '''Existing table columns: ['text', 'headline', 'text_emotion']'''  
            + '''. The detailed description for each existing column: 'text' column contains the posts to be analyzed; 'text_emotion' column provides emotion identified from the 'text' column. IMPORTANT: emotion values of the 'text_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'; '''
            + '''. The detailed description for the new column to be added: 'headline_emotion' column provides emotion identified from the 'headline' column. IMPORTANT: emotion values of the 'headline_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'. '''
        },
        {
            "role": "assistant",
            "content": "False##The new column contains emotion from the 'headline' column, while the existing column contains emotion from the 'text' column. Thus, the description for the new column does not match any of the existing columns.",
        },
        {
            "role": "user",
            "content": '''Existing table columns: ['text', 'emotion']'''  
            + '''. The detailed description for each existing column: 'text' column contains the posts to be analyzed; 'emotion' column provides emotion identified from the 'text' column. IMPORTANT: emotion values of the 'emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'; '''
            + '''. The detailed description for the new column to be added: 'text_emotion' column provides emotion identified from the 'text' column. IMPORTANT: emotion values of the 'text_emotion' column can only be either 'anticipation', 'anger', 'fear', 'sadness', 'joy', 'trust', or 'disgust'. '''
        },
        {
            "role": "assistant",
            "content": "True#emotion#The new column 'text_emotion' to be added contains exactly the same information as the existing 'emotion' column.",
        },
        {
            "role": "user",
            "content": '''Existing table columns: ''' + str(enum) 
            + '''. The detailed description for each existing column: ''' + all_description 
            + '''. The detailed description for the new column to be added: ''' + new_description
        }
    ]

    if gpt4:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-4-0613",
            messages=messages,
        )
        if verbose:
            # num_token_msg = num_tokens_from_messages(messages, "gpt-4-0613")
            # print("Number of tokens of messages for 'check_alias': ", num_token_msg)
            print("VERBOSE:"+"Number of prompt tokens for 'check_alias': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'check_alias': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'check_alias': ", response.usage.total_tokens)
            current_pize = 0.00003 * response.usage.prompt_tokens + 0.00006 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'check_alias': ${current_pize}")
            autopipeline.cost += current_pize
            autopipeline.table_gen_cost += response.usage.prompt_tokens
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")
    else:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )
        if verbose:
            # num_token_msg = num_tokens_from_messages(messages, "gpt-3.5-turbo-0125")
            # print("Number of tokens of messages for 'check_alias': ", num_token_msg)
            print("VERBOSE:"+"Number of prompt tokens for 'check_alias': ", response.usage.prompt_tokens)
            print("VERBOSE:"+"Number of answer tokens for 'check_alias': ", response.usage.completion_tokens)
            print("VERBOSE:"+"Number of total tokens for 'check_alias': ", response.usage.total_tokens)
            current_pize = 0.0000005 * response.usage.prompt_tokens + 0.0000015 * response.usage.completion_tokens
            print("VERBOSE:"+f"Cost for 'check_alias': ${current_pize}")
            autopipeline.cost += current_pize
            autopipeline.table_gen_cost += response.usage.prompt_tokens
            print("VERBOSE:"+f"Accumulated cost: ${autopipeline.cost}")

    res = response.choices[0].message.content
    if verbose:
        print("VERBOSE:"+"Check alias output: ", res)
        print("VERBOSE:"+"Check alias prompt cost: ", autopipeline.table_gen_cost)
    res_ls = res.split("#")
    if res_ls[0] != "True":
        return ""
    return res_ls[1]

def register_func(name, user_function):
    autopipeline._callbacks[name] = user_function

def format(x):
    if pd.api.types.is_float_dtype(x):
        return x.round(3).astype(str)
    return x.astype(str)

def evaluation(result, answer):

    # dealing with series
    try:
        result = result.to_frame()
    except:
        pass

    for df in answer:
        try:
            result = result.fillna(0) 
            df = df.fillna(0) 

            df = df.apply(format)
            result = result.apply(format)

            result = result.iloc[:, result.iloc[0].argsort()]
            df = df.iloc[:, df.iloc[0].argsort()]

            are_values_equal = (result.to_numpy() == df.to_numpy()).all()

            if are_values_equal:
                return True
        except:
            try: 
                if result == df: # for numerical values and lists
                    return True
                if (result == df).all(): # for numerical values and lists
                    return True
                if np.isclose(result, df): # for numerical values and lists
                    return True
            except:
                pass
    return False

def ensure_max_words(text, max_words=20):
    words = str(text).split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'  # Add ellipsis after truncation
    return text

