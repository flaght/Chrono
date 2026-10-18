def get_dates(method):
    if method == 'test0':
        return '2026-07-01', '2026-08-13'
    elif method == 'test1':
        return '2026-06-01', '2026-08-13'
    elif method == 'train0':
        return '2025-06-01', '2026-01-01'


def get_ranges(method):
    if method == 'train0':
        train_range = ('2025-06-01', '2025-10-31')
        val_range = ('2025-11-01', '2026-01-01')
        test_range = ('2026-07-01', '2026-08-13')
        return train_range, val_range, test_range
    elif method == 'test0':
        train_range = ('2026-07-01', '2026-08-13')
        val_range = ('2026-07-01', '2026-08-13')
        test_range = ('2026-07-01', '2026-08-13')
        return train_range, val_range, test_range
