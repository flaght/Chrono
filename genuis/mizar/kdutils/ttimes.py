def get_dates(method):
    if method == 'aicso0':
        return '2020-01-01', '2025-03-15'
    elif method == 'bicso0':
        return '2020-01-01', '2025-03-15'
    elif method == 'bicso1':
        return '2025-01-01', '2025-09-17'
    elif method == 'bicso2':
        return '2011-01-01', '2025-09-17'
    elif method == 'bicso3':
        return '2015-01-01', '2026-05-10'
    elif method == 'cicso0':
        return '2022-07-25', '2025-03-13'
    elif method == 'cicso1':
        return '2020-01-02', '2025-08-29'
    elif method == 'cicso2':
        return '2020-01-01', '2026-01-01'
    elif method == 'dicso2':
        return '2011-01-01', '2025-09-17'
    elif method == 'ricso2':
        return '2012-01-01', '2026-04-30'
    elif method == 'ricso3':
        return '2015-01-01', '2026-04-30'
    elif method == 'ricso4':
        return '2014-07-01', '2026-06-30'
    elif method == 'ticso1':
        return '2026-05-06', '2026-07-09'


FIIXED_MAPPING = {
    'ricso2': {
        'train_end': '2022-04-30 15:00:00',
        'val_end': '2025-04-30 21:00:00',
        'recent_start': '2021-03-25 21:00:00'
    }
}
