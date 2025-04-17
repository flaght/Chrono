import os

## brief

#codes = ['BU', 'FU', 'HC', 'L', 'M', 'MA', 'PP', 'RB', 'SR', 'TA', 'V']

instruments_codes = {
    'brief': ['BU', 'FU', 'HC', 'L', 'M', 'MA', 'PP', 'RB', 'SR', 'TA', 'V'],
    'istocks': ['IF', 'IM', 'IC', 'IH'],
    'ifs': ['IF'],
    'ihs': ['IH'],
    'ics': ['IC'],
    'ims': ['IM'],
    'rbb': ['RB']
}

base_path = os.path.join('/workspace/data/dev/kd/evolution/lumina',
                         os.environ['LUMINA_BASE_NAME'])

codes = instruments_codes[os.environ['INSTRUMENTS']]