import os
import json
import random
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, default='/mydata/data/seunghoonjeong/co3dv2_single_preprocess') 
    parser.add_argument("--seed", type=int, default=42)
    return parser

def main():
    args = get_parser().parse_args()
    random.seed(args.seed)

    # Load the dataset
    with open(os.path.join(args.preprocess_dir, 'paired_data_train.json'), 'r') as f:
        train_dataset = json.load(f)

    with open(os.path.join(args.preprocess_dir, 'paired_data_test.json'), 'r') as f:
        test_dataset = json.load(f)
    
    category_list = list(train_dataset.keys())

    train_seq_name = {}
    test_seq_name = {}
    refined_train_dataset = {}
    refined_test_dataset = {}
    refined_train_dataset_size = 0
    for category in category_list:
        train_data = train_dataset[category]
        test_data = test_dataset[category]
        all_data = train_data + test_data

        seq_name_list = list(set([data['seq_name'] for data in all_data]))
        random.shuffle(seq_name_list)

        train_seq_name_list = seq_name_list[:int(len(seq_name_list)*0.9)]
        test_seq_name_list = seq_name_list[int(len(seq_name_list)*0.1):]

        refined_train_data = [data for data in all_data if data['seq_name'] in train_seq_name_list]
        refined_test_data = [data for data in all_data if data['seq_name'] in test_seq_name_list]

        for data in refined_train_data:
            data["original_split"] = "train"
        
        for data in refined_test_data:
            data["original_split"] = "test"

        train_seq_name[category] = train_seq_name_list
        test_seq_name[category] = test_seq_name_list

        refined_train_dataset[category] = refined_train_data
        refined_test_dataset[category] = refined_test_data
        refined_train_dataset_size += len(refined_train_data)

    print(f"Train dataset size: {refined_train_dataset_size}")

    with open(os.path.join(args.preprocess_dir, 'refined_paired_data_train.json'), 'w') as f:
        json.dump(refined_train_dataset, f, indent=4)
    

    with open(os.path.join(args.preprocess_dir, 'refined_paired_data_test.json'), 'w') as f:
        json.dump(refined_test_dataset, f, indent=4)

    with open(os.path.join(args.preprocess_dir, 'train_seq_name.json'), 'w') as f:
        json.dump(train_seq_name, f, indent=4)

    with open(os.path.join(args.preprocess_dir, 'test_seq_name.json'), 'w') as f:
        json.dump(test_seq_name, f, indent=4)

if __name__ == '__main__':
    main()
