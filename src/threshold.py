import json
import os

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Script to threshold the pseudo-labels', add_help=add_help)

    # Dataset
    parser.add_argument('--input-file')
    parser.add_argument('--output-file')
    parser.add_argument('--parametrization', default=3, type=int,
                        help='parametrization to use, choose between 1, 2 or 3 (default: 1)')
    parser.add_argument('--tl', default=0.9, type=float,
                        help='Value for tau_low (default: 0.9')
    parser.add_argument('--th', default=1., type=float,
                        help='Value for tau_high (default: 1.)')


    return parser

def parametrization_1(file, t_h):
    data = json.load(open(file))

    new_dict = {}
    first = True
    # Iterate over all data (union of labeled and pseudo-labeled)
    for i in data:
        # If we have a pseudo-labeled data
        if 'scores' in data[i]:
            # Loop over all predictions
            for b, s in zip(data[i]['bbox'], data[i]['scores']):
                # If score of current prediction >= tau_h, we add the prediction  
                if s >= t_h:
                    if first:
                        new_dict[i] = {}
                        new_dict[i]['bbox'] = [b]
                        new_dict[i]['scores'] = [1]
                        first = False
                    else:
                        new_dict[i]['bbox'].append(b)
                        new_dict[i]['scores'].append(1)
            first = True
        # If we have a labeled data, we copy the initial boxes
        else:
            new_dict[i] = data[i]

    return new_dict

def parametrization_2(file, t_l, t_h):
    data = json.load(open(file))

    new_dict = {}
    first = True
    # Iterate over all data (union of labeled and pseudo-labeled)
    for i in data:
        # If we have a pseudo-labeled data
        if 'scores' in data[i]:
            # If the maximum score of the current pseudo-label < tau_h, there is no FG example -> skip
            if max(data[i]['scores'], default=0) < t_h and len(data[i]['scores']) > 0:
                continue
            # Loop over all predictions
            for b, s in zip(data[i]['bbox'], data[i]['scores']):
                # If score of current prediction >= tau_h, we add the prediction
                if s >= t_h:
                    if first:
                        new_dict[i] = {}
                        new_dict[i]['bbox'] = [b]
                        new_dict[i]['scores'] = [1]
                        first = False
                    else:
                        new_dict[i]['bbox'].append(b)
                        new_dict[i]['scores'].append(1)
                # If score of current prediction is: tau_l =< prediction < tau_h, the prediction will not contribute to the loss
                # because we set its class to -2 -> ignored by sampler
                elif s < t_h and s >= t_l:
                    b[-1] = -2 
                    if first:
                        new_dict[i] = {}
                        new_dict[i]['bbox'] = [b]
                        new_dict[i]['scores'] = [s]
                        first = False
                    else:
                        new_dict[i]['bbox'].append(b)
                        new_dict[i]['scores'].append(s)
            first = True
        # If we have a labeled data, we copy the initial boxes
        else:
            new_dict[i] = data[i]

    return new_dict

def parametrization_3(file, t_l, t_h):
    data = json.load(open(file))

    new_dict = {}
    first = True
    # Iterate over all data (union of labeled and pseudo-labeled)
    for i in data:
        # If we have a pseudo-labeled data
        if 'scores' in data[i]:
            # If the maximum score of the current pseudo-label < tau_l, there is no FG example -> skip
            if max(data[i]['scores'], default=0) < t_l and len(data[i]['scores']) > 0:
                continue
            # Loop over all predictions
            for b, s in zip(data[i]['bbox'], data[i]['scores']):
                # If score of current prediction >= tau_h, we add the prediction and its score = 1
                if s >= t_h:
                    if first:
                        new_dict[i] = {}
                        new_dict[i]['bbox'] = [b]
                        new_dict[i]['scores'] = [1]
                        first = False
                    else:
                        new_dict[i]['bbox'].append(b)
                        new_dict[i]['scores'].append(1)
                # If score of current tau_l =< prediction < tau_h, we use the score to weight the loss
                elif s < t_h and s >= t_l:
                    if first:
                        new_dict[i] = {}
                        new_dict[i]['bbox'] = [b]
                        new_dict[i]['scores'] = [s]
                        first = False
                    else:
                        new_dict[i]['bbox'].append(b)
                        new_dict[i]['scores'].append(s)
            first = True
        # If we have a labeled data, we copy the initial boxes
        else:
            new_dict[i] = data[i]

    return new_dict

def main(args):

    print(args)

    tau_low = args.tl
    tau_high = args.th
    parametrization = args.parametrization
    input_file = args.input_file
    output_file = args.output_file

    if parametrization == 1:
        new_data = parametrization_1(input_file, tau_high)
    elif parametrization == 2:
        new_data = parametrization_2(input_file, tau_low, tau_high)
    elif parametrization == 3:
        new_data = parametrization_3(input_file, tau_low, tau_high)
    else:
        raise ValueError('Invalid parametrization')

    json.dump(new_data, open(output_file, 'w'))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
