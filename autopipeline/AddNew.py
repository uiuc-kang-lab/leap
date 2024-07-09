import re
from .util import formalize_desc, check_alias
import autopipeline

def wrapper(callback_name, input_vars, para_dict, desc_dict, enum, description, verbose, dot, client, gpt4):

    callback = autopipeline._callbacks[callback_name]
    if not (callback and callable(callback)):
        if verbose:
            print("VERBOSE:"+"Function " + callback_name+ " undefined.")
        return table, enum, description, dot

    # check_alias
    output_dict = {}
    desc_dict_new = {}
    i = 1
    # sort the dictionary
    desc_dict = {k: desc_dict[k] for k in sorted(desc_dict)}
    for key, value in desc_dict.items():
        output_dict[key] = callback_name+"_output"+str(i)
        desc_dict_new[callback_name+"_output"+str(i)] = value
        input_vars.append(callback_name+"_output"+str(i))
        i += 1
    new_description = formalize_desc(desc_dict_new)
    matches = re.findall(r"\[(.*?)\]", new_description)
    para_ls = []
    for match in matches:
        para_ls.append(para_dict[match])

    replacement_iter = iter(para_ls)

    # Function to get the next replacement value
    def replacement(match):
        return next(replacement_iter, '')

    # Use re.sub with a function as the replacement
    new_description = re.sub(r"\[.*?\]", replacement, new_description)

    col = check_alias(enum, description, new_description, verbose, client, gpt4)
    if len(col) > 0:
        for para_key in para_dict:
            dot.edge(para_dict[para_key], col, callback_name)
        return table, enum, description, dot

    table = callback(*input_vars)
    enum = list(table.columns)
    description += " " + new_description
    new_cols = list(desc_dict.keys())
    for new_col in new_cols:
        new_col = output_dict[new_col]
        for para_key in para_dict:
            dot.edge(para_dict[para_key], new_col, callback_name)

    return table, enum, description, dot
