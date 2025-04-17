def get_dates(method):
    if method == 'train':
        return '2018-01-01', '2023-01-01'
    elif method == 'val':
        return '2023-01-01', '2024-01-01'
    elif method == 'test':
        return '2024-01-01', '2024-07-01'
    elif method == 'micro':
        return '2022-07-22', '2024-10-30'
    elif method == 'nicso':
        return '2020-01-01', '2024-10-30'
    elif method == 'mini':
        return '2023-06-01', '2023-06-10'
    elif method == 'exper':
        return '2023-06-01', '2024-01-01'