import argparse

from utils.helper import read_jsonl, get_classification_report
from collections import defaultdict


def train(data_path):
    """Train most-frequent-tag baseline.

    Args:
        data_path (str): path to training data in JSON lines format.

    Returns:
        Callable: a function to predict tags based on learned parameters.
    """
    print('> Training model on dataset', data_path)
    sample_count = 0
    token_frequent_dic = defaultdict(dict)
    tag_frequent_dic = defaultdict(int)
    for sample in read_jsonl(data_path):
        sample_count += 1
        tokens = sample['tokens']
        tags = sample['tags']
        print(tokens)
        print(tags)
        # '''
        # YOUR CODE HERE
        # '''
        for i in range(len(tokens)):
            if tokens[i] not in token_frequent_dic:
                token_frequent_dic[tokens[i]] = defaultdict(int)
            token_frequent_dic[tokens[i]][tags[i]] += 1
            tag_frequent_dic[tags[i]] += 1
    for token in token_frequent_dic:
        tag_list = token_frequent_dic[token]
        max_token = max(tag_list, key=tag_list.get)
        token_frequent_dic[token] = {max_token: tag_list[max_token]}
    max_tag = max(tag_frequent_dic, key=tag_frequent_dic.get)
    # return token_frequent_dic, most_common_tag

    # '''
    # Set model model params to be a dictionary with tag frequency counts
    # '''
    model_params = {
        # YOUR CODE HERE
        'known_token':token_frequent_dic,
        'unknown_token': {max_tag: tag_frequent_dic[max_tag]}
    }
    # return model_params

    print(f'> Finished training on {sample_count} samples')
    return lambda x: predict(x, model_params)


def predict_tag(token, params):
    """Return most frequent tag for `token`.

    If `token` is unkown, return most frequent tag in the training data. 

    Args:
        token (str): the token string.
        params (Dict): parameters containing the most frequent tags for each token.

    Returns:
        Tuple[str, int]: a tuple containing the predicted tag and its correponding frequency.
    """
    
    '''
    YOUR CODE HERE
    '''
    if token in params['known_token']:
        tag_pred = list(params['known_token'][token].keys())[0]
        tag_count = params['known_token'][token][tag_pred]
    else:
        tag_pred = list(params['unknown_token'].keys())[0]
        tag_count = 0
    # pass
    return tag_pred, tag_count


def predict(x, params):
    """Predict tags for inputs `x`, using learned parameters `params`.

    Args:
        x (Union[str, List[str], List[List[str]]]): the input data. It can be a single string, 
        a list of strings, or a list of list of tokens.
        params (Dict): parameters containing the most frequent tags for each token.

    Returns:
        Union[List[List[Tuple[str, str, int]]], List[Tuple[str, str, int]]]: predicted tags 
        for each token as tuples (token, tag, count).
    """
    is_string = type(x) == str
    if is_string:
        x = [x]

    preds = []
    for sample in x:
        sample_preds = []

        if type(sample) == str:
            # note that this may not match the dataset tokenization
            tokens = sample.split()
        else:
            # this sample is already tokenized
            tokens = sample
        
        for token in tokens:
            prediction, count = predict_tag(token, params)
            sample_preds.append((token, prediction, count))

        preds.append(sample_preds)
    
    # for convenience, unpack result when input is a single string
    if is_string: 
        preds = preds[0]
    return preds


def evaluate(model, data_path):
    """Evaluate model performance on a dataset

    Args:
        model (Any): [description]
        data_path (str): path to dataset in JSON lines format
    """
    predictions = []
    labels = []

    print('> Evaluating model on dataset', data_path)

    for sample in read_jsonl(data_path):
        sample_preds = model([sample['tokens']])[0]

        for pred, tag in zip(sample_preds, sample['tags']):
            predictions.append(pred[1])
            labels.append(tag)
        
    # print()
    print("Full classification report:")
    report = get_classification_report(labels, predictions, zero_division=0, digits=3)
    print(report)

    print("\nClassification report for top tag classes (PLEASE REPORT THIS ONE):")
    report = get_classification_report(labels, predictions, zero_division=0, digits=3, top_k=6)
    print(report)
    

def check_samples(model):
    """Check the model predictions

    Args:
        model (Any): a trained model
    """
    samples = ['This is a simple model .', 'I love NLP !']
    expected = [[('This', 'PRON', 109), ('is', 'AUX', 1872), ('a', 'DET', 3589), ('simple', 'ADJ', 19), ('model', 'NOUN', 15), ('.', 'PUNCT', 8640)],
                [('I', 'PRON', 3121), ('love', 'VERB', 54), ('NLP', 'NOUN', 0), ('!', 'PUNCT', 529)]]

    error_msg = '\nPredictions for sample "{}" do not match expected values: \n{} \nPlease check your implementation.'
    for idx, sample in enumerate(samples):
        pred = model(sample)
        assert pred == expected[idx], error_msg.format(sample, expected[idx])

    print('> All sample checks passed.')


def run(train_path, validation_path):
    print()
    model = train(train_path)
    check_samples(model)
    
    if validation_path:
        evaluate(model, validation_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', help='path to training dataset in JSON lines format')
    p.add_argument('--validation_path', help='path to validation dataset in JSON lines format')
    args = p.parse_args()
    run(args.train_path, args.validation_path)

