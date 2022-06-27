import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from captum.attr import LayerIntegratedGradients
import argparse
import pickle

# parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument('--path_train', help='Specify file path to the train set.', type=str, required=True)
parser.add_argument('--path_valid', help='Specify file path to the validation set.', type=str, required=True)
parser.add_argument('--path_test', help='Specify file path to the test set.', type=str, required=True)
parser.add_argument('--path_ckpt', help='Specify file path to the model checkpoint.', type=str, required=True)
parser.add_argument('--path_save', help='Specify path the output folder.', type=str, required=True)
parser.add_argument('--model_name', help='Specify the model name', type=str, required=True)
parser.add_argument('--target', help='Supported target variables ["sic2", "sic4"].', type=str, default="sic2")
parser.add_argument('--max_seq_length', help='Specify maximum sequence length.', type=int, default=512)
parser.add_argument('--batch_size', help='Specify batch size for the dataloader.', type=int, default=4)
parser.add_argument('--internal_batch_size', help='Specify internal batch size for the integrated gradients.', type=int, default=4)
args=parser.parse_args()

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prevents duplicate library import error 
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read_data(path_train, path_valid, path_test, target):
    # read data
    train = pd.read_parquet(path_train)
    valid = pd.read_parquet(path_valid)
    test = pd.read_parquet(path_test)
    num_labels = len(np.unique(train[target]))
    return train,valid,test,num_labels

# Define pretrained tokenizer and model
def init_model(model_name, path_ckpt, num_labels, **kwargs):
    if model_name.startswith("bert") or model_name.startswith("Prosus"):
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(path_ckpt, num_labels=num_labels, ignore_mismatched_sizes=True)
        embed = model.bert.embeddings
    elif model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(path_ckpt, num_labels=num_labels)
        embed = model.roberta.embeddings
    elif model_name.startswith("custom"):
        vocab_path = f"{path_ckpt}/vocab.txt"
        config_path = f"{path_ckpt}/config.json"
        model_path = f"{path_ckpt}/pytorch_model.bin"
        tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(model_path, config=config_path, num_labels=num_labels)
        embed = model.bert.embeddings
    
    return model, tokenizer, embed

# BERT Dataset class
class ECData(Dataset):
    def __init__(self, X,y, tokenizer, max_length=512):
        super().__init__()

        self.X = X
        self.y = y
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        # get batch
        X = self.X[index]
        y = self.y[index]

        # process text
        encoded = self.tokenizer.encode_plus(X, return_attention_mask=True, return_token_type_ids=False,padding='max_length', max_length=self.max_length, truncation=True, add_special_tokens=True, return_tensors='pt')

        # position ids
        encoded["position_ids"] = torch.arange(self.max_length, dtype=torch.long).unsqueeze(0).expand_as(encoded["input_ids"])
        encoded["ref_position_ids"] =  torch.zeros(self.max_length, dtype=torch.long).unsqueeze(0).expand_as(encoded["input_ids"])

        # ref inputs
        ref_input_ids = torch.zeros(self.max_length, dtype=torch.long)
        ref_input_ids[0] = encoded["input_ids"][0][0]
        ref_input_ids[-1] = encoded["input_ids"][0][-1]
        encoded["ref_input_ids"] = ref_input_ids.unsqueeze(0).expand_as(encoded["input_ids"])
        
        for k,v in encoded.items(): encoded[k] = v.squeeze(0)
        
        return encoded["input_ids"], encoded["attention_mask"], encoded["position_ids"], encoded["ref_position_ids"], ref_input_ids, torch.as_tensor(y, dtype=torch.int64), 


def main():
    X_train, X_valid, X_test, num_labels = read_data(args.path_train, args.path_valid, args.path_test, args.target)
    model, tokenizer, embed = init_model(args.model_name, args.path_ckpt, num_labels)

    # setup model
    model.to(device)
    model.eval()
    model.zero_grad()

    def model_forward_wrapper(*args):
        output = model(*args)
        return output.logits

    # if args.model_name == "custom":
    #     args.path_save = args.path_save + "/" + args.path_ckpt.split("/")[-2]
    print(f"Saving data to {args.path_save}...")
    
    # Layer Integrated Gradients
    lig = LayerIntegratedGradients(model_forward_wrapper, embed)
    
    data_test  = ECData(X=X_test["text"],y=X_test[args.target], tokenizer=tokenizer, max_length=args.max_seq_length)
    dl_test    = DataLoader(dataset=data_test, batch_size=args.batch_size, shuffle=False)

    all_attr = torch.tensor([], dtype=torch.float)
    all_delta = torch.tensor([], dtype=torch.float)
    all_pred = torch.tensor([], dtype=torch.float)

    for (input_ids, attention_mask, position_ids, ref_position_ids, ref_input_ids, y_true) in tqdm(dl_test):
        # place batch on device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        ref_position_ids = ref_position_ids.to(device)
        ref_input_ids = ref_input_ids.to(device)

        # get predictions
        out = model_forward_wrapper(input_ids, attention_mask)
        pred = torch.argmax(out, axis=-1)
        attr, delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids, additional_forward_args = attention_mask,return_convergence_delta=True, target=pred, internal_batch_size = args.internal_batch_size)

        # fix last batch size
        if attr.dim() == 1:
            attr = attr.unsqueeze(0)
            delta = delta.unsqueeze(0)
            pred = pred.unsqueeze(0)

        # vector norm
        attr = attr.sum(dim=-1)
        attr = attr / torch.norm(attr)

        # save
        all_attr = torch.cat((all_attr, attr.cpu()))
        all_delta = torch.cat((all_delta, delta.cpu()))
        all_pred = torch.cat((all_pred, pred.cpu()))
    
    with open(f'{args.path_save}.p', 'wb') as handle:
            pickle.dump({"attributions":all_attr.numpy(), "delta":all_delta.numpy(), "predictions":all_pred.numpy()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main()