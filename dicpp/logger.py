def msg(text, level=0):
    print('\t'*level + text)
    return text

def warn(text, level=0):
    text = 'WARNING: ' + text
    print('\t'*level + text)
    return text

def error(text, level=0):
    text = 'ERROR: ' + text
    print('\t'*level + text)
    return text
