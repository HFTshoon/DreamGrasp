import os
import json
import random
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, default='/mydata/data/seunghoonjeong/DTU_single_preprocess') 
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
    
    scan_list = list(train_dataset.keys())
    random.shuffle(scan_list)
    train_scan_list = scan_list[:int(len(scan_list)*0.9)]
    test_scan_list = scan_list[int(len(scan_list)*0.1):]

    train_seq_name = {}
    test_seq_name = {}
    refined_train_dataset = {}
    refined_test_dataset = {}
    refined_train_dataset_size = 0
    for scan in scan_list:
        train_data = train_dataset[scan]
        for data in train_data:
            data["original_split"] = "train"

        test_data = test_dataset[scan]
        for data in test_data:
            data["original_split"] = "test"

        all_data = train_data + test_data
        seq_name_list = list(set([data['seq_name'] for data in all_data]))

        if scan in train_scan_list:
            train_seq_name[scan] = seq_name_list
            refined_train_dataset[scan] = all_data
            refined_train_dataset_size += len(all_data)
        else:
            test_seq_name[scan] = seq_name_list
            refined_test_dataset[scan] = all_data

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
