from .utils import invert_dict, merge_dict

short_labels = {
    1: 'VL',
    2: 'VM',
    3: 'VI',
    4: 'RF',
    5: 'SAR',
    6: 'GRA',
    7: 'AM',
    8: 'SM',
    9: 'ST',
    10: 'BFL',
    11: 'BFS',
    12: 'AL'
}

long_labels = {
    1: 'Vastus Lateralis',
    2: 'Vastus Medialis',
    3: 'Vastus Intermedius',
    4: 'Rectus Femoris',
    5: 'Sartorius',
    6: 'Gracilis',
    7: 'Adductor Magnus',
    8: 'Semimembranosus',
    9: 'Semitendinosus',
    10: 'Biceps Femoris Long',
    11: 'Biceps Femoris Short',
    12: 'Adductor Longus'
}

inverse_labels = merge_dict(invert_dict(short_labels), invert_dict(long_labels))