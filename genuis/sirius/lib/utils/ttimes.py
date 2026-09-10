def get_dates(method):
    if method == 'bisco1':
        return '2025-04-01', '2025-04-30'
    if method == 'ricso2':
        return '2012-01-01', '2026-04-30'
    elif method == 'ricso3':
        return '2025-03-01', '2026-04-30'


FIIXED_MAPPING = {
    'ricso2': {
        'train_end': '2022-04-30 15:00:00',
        'val_end': '2025-04-30 21:00:00',
        'recent_start': '2021-03-25 21:00:00'
    }
}
