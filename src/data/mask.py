from random import random

def create_attention_mask(sentence, ratio=0.35):
    # sentence is a list that contain word from a split sentence
    masked_data = []
    mask_index = []
    for word in sentence:
        if random() > ratio:
            masked_data.append(word)
            mask_index.append(1)
        else:
            masked_data.append('<unk>')
            mask_index.append(0)
    return masked_data, mask_index, sentence

