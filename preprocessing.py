def flatten(list_of_lists):
    """
    Flattens from two-dimensional list to one-dimensional list
    """
    return [item for sublist in list_of_lists for item in sublist]


def adjust_sequences(data, length=50):
    """
    Fixes the input and output sequences length, adding padding or truncating if necessary
    :param data json file containing entries from atis dataset.
    :param length the fixed length of the sentence.
    """
    for sample in data['data']:
        # adjust the sequence of input words
        if len(sample['words']) < length:
            # add <EOS> and <PAD> if sentence is shorter than maximum length
            sample['words'].append('<EOS>')
            while len(sample['words']) < length:
                sample['words'].append('<PAD>')
        else:
            # otherwise truncate and add <EOS> at last position
            sample['words'] = sample['words'][:length]
            sample['words'][-1] = '<EOS>'

        # adjust in the same way the sequence of output slots
        if len(sample['slots']) < length:
            sample['slots'].append('<EOS>')
            while len(sample['slots']) < length:
                sample['slots'].append('<PAD>')
        else:
            sample['slots'] = sample['slots'][:length]
            sample['slots'][-1] = '<EOS>'

    return data

def get_vocabularies(train_data):
    """
    Collect the input vocabulary, the slot vocabulary and the intent vocabulary
    :param train_data the training data containing words,slots and intent.
    """
    # from a list of training examples, get three lists (columns)
    data = train_data['data']
    seq_in = [sample['words'] for sample in data]
    vocab = flatten(seq_in)
    # removing duplicated but keeping the order
    v = ['<PAD>', '<SOS>', '<EOS>'] + vocab
    vocab = sorted(set(v), key=lambda x: v.index(x)) # https://docs.python.org/3.3/howto/sorting.html
    s = ['<PAD>','<EOS>'] + train_data['meta']['slot_types']
    slot_tag = sorted(set(s), key=lambda x: s.index(x))
    i = train_data['meta']['intent_types']
    intent_tag = sorted(set(i), key=lambda x: i.index(x))

    return vocab, slot_tag, intent_tag

def create_mappings(vocab,forward_map):
    """
    This function takes the words in the vocabulary and creates a unique mapping to a number.
    :param vocab contains all the words in the corpus.
    :param forward_map a dictionary that will be populated with mappings.
    returns populated forward_map
    """
    for sample in vocab:
        if sample not in forward_map.keys():
            forward_map[sample]= len(forward_map)
    return forward_map

def prepare_sequence(seq_data, mapping,map_type):
    """
    :param seq a sequnce which will be embedded as a vector
    :param mapping, a dictionary which contains how each element in the seq will be mapped to a number.
    :param map_type 'words','slots' or 'intent'
    returns a Pytorch Tensor.
    """
    if map_type=='intent':
        intent = seq_data[map_type]
        embeddings = mapping[intent] if intent in mapping.keys() else mapping["<UNK>"]
        return torch.tensor(embeddings)
    else:
        embed_fnc = lambda word: mapping[word] if word in mapping.keys() else mapping["<UNK>"]
        embeddings = list(map(embed_fnc, seq_data[map_type]))
        return torch.LongTensor(embeddings)

def create_training_set(padded_atis):
    """
    :param padded_atis, this is padded sequence data.
           Of the form seq,slots,intent. This function coverts
           these into tensors.
    return train_data; [(seq_tensor,slot_tensor,intent_tensor)]
    """
    train_data = []
    atis_data = padded_atis['data']
    for i in range(len(atis_data)):
        seq_tensor = prepare_sequence(atis_data[i],word2index,'words')
        slot_tensor = prepare_sequence(atis_data[i],tag2index,'slots')
        intent_tensor = prepare_sequence(atis_data[i],intent2index,'intent')
        train_data.append((seq_tensor,slot_tensor,intent_tensor))
    return train_data

def concatenate_batch(batch):
    seqs = torch.stack([ex[0] for ex in batch])
    slots = torch.stack([ex[1] for ex in batch])
    intents = torch.stack([ex[2] for ex in batch])
    return seqs,slots,intents

def get_batches(batch_size, train_data):
    """
    Returns iteratively a batch of specified size on the data.
    The last batch can be smaller if the total size is not multiple of the batch
    """
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while sindex < len(train_data):
        batch = train_data[sindex:eindex] #list of batch_size num of tuples.
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        #print('returning', len(batch), 'samples')
        yield concatenate_batch(batch)
