import math, uuid

def string_to_int(string, alphabet):
    number = 0
    alpha_len = len(alphabet)
    for char in string[::-1]:
        number = number * alpha_len + alphabet.index(char)
    return number


def init_bet():
    alphabet = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
                    "abcdefghijkmnopqrstuvwxyz")
    new_alphabet = list(sorted(set(alphabet)))
    if len(new_alphabet) > 1:
        alphabet = new_alphabet
    return alphabet


def create_id(original=None, digit=8):
    ori_str = original if original is not None else uuid.uuid1().hex[-12:]
    s = string_to_int(ori_str, init_bet())
    return str(int(math.pow(10, digit - 1) + abs(hash(s)) % (10**(digit - 2))))