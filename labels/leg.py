from .utils import invert_dict, merge_dict

short_labels = {
    1: 'SOL',
    2: 'GM',
    3: 'GL',
    4: 'TA',
    5: 'ELD',
    6: 'PE',
}

long_labels = {
    1: 'Soleus',
    2: 'Gastrocnemius Medialis',
    3: 'Gastrocnemius Lateralis',
    4: 'Tibialis Anterior',
    5: 'Extensor Longus Digitorum',
    6: 'Peroneus'
}

inverse_labels = merge_dict(invert_dict(short_labels), invert_dict(long_labels))
