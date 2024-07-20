import pandas as pd
import os
import autopipeline
import time
from openai import OpenAI
from .PipelineGen import pipeline_gen
from .Filter import check_vague
from .util import register_func

def leap(query, table_ptr, description, verbose = False, udf = None, saving_mode=False):
    autopipeline.cost = 0

    if saving_mode == False:
        gpt4 = True
    else:
        gpt4 = False

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    client = OpenAI(api_key=autopipeline.api_key, organization=autopipeline.organization)

    table = table_ptr.copy()
    columns = str(table.columns.tolist())

    print("********** Checking Query...")
    if udf == None:
        precheck, functions = check_vague(query, columns, description, verbose, client)
    else:
        func_desc = ""
        for f in udf:
            f['func-name'] = f["func"].__name__
        for f in udf:
            func_desc += (f['func-name'] + " : " + f['func-description'] + "\n")
            register_func(f["func-name"], f["func"])
        precheck, functions = check_vague(query, columns, description, verbose, client, func_desc)

    precheck = precheck.content
    if verbose:
        print("########## Precheck Result:", precheck)
        print("\n")
    ls = precheck.split('#')
    res = ls[0]
    hint = ls[1]
    if res == "False":
        print("########## Feedback: ", hint + " Please be more detailed on your query.")
        if verbose: # print functions only when verbose
            print("\nThe supported function list is as follows: \n" + functions)
        return None, table
    if "WARNING" in hint:
        warning_msg = "WARNING: "+hint.split("WARNING:")[1].strip()
        print("########## "+warning_msg)
        user_reponse = str(input("Proceed? [Y/n]"))
        if user_reponse == "Y":
            hint = hint.split("WARNING")[0].strip()
        else:
            return None, table
    print("********** Query Check Pass!")

    require_new, feedback, result, table = pipeline_gen(query, table, hint, description, verbose, client, udf, gpt4)
    if require_new:
        print("########## Feedback: ", feedback + " Please be more detailed on your query.")
    else:
        print("Succeed!")
    if verbose:
        print(f"Total cost: ${autopipeline.cost}")
    return result, table