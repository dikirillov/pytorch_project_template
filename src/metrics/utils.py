# Based on seminar materials

# Don't forget to support cases when target_text == ''

import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if target_text == "":
        return int(predicted_text != "")
    return editdistance.eval(list(target_text), list(predicted_text)) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        return int(predicted_text != "")
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(
        target_text.split()
    )
