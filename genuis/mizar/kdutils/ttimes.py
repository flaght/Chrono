def get_dates(method):
    if method == 'aicso0':
        return '2020-01-01', '2025-03-15'
    elif method == 'aicso1':
        return '2015-01-01', '2025-01-01'
    elif method == 'aicso2':
        return '2015-01-01', '2025-04-10'
    elif method == 'aicso3':
        return '2019-01-01', '2025-05-01'
    
    if method == 'bicso0':
        return '2020-01-01', '2025-03-15'

    if method == 'bicso1':
        return '2025-01-01', '2025-09-17'