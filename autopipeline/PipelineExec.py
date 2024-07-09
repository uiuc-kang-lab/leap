def query_exec(table, code, status, verbose):
    print(code)
    # print("\n")
    namespace = {'res': 0, 'table': table}
    try:
        exec(code, namespace)
    except:
        feedback = "error occurred during execution" # maybe the error message?
        return namespace['res'], status, True, feedback
    status.append('code executed')
    return namespace['res'], status, False, ""

def display(result, status, verbose):
    print(result)
    status.append('displayed')
    return status