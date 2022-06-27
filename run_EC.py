import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch 
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import warnings
import wandb
import argparse

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument('--path_train', help='Specify file path to the train set.', type=str, required=True)
parser.add_argument('--path_valid', help='Specify file path to the validation set.', type=str, required=True)
parser.add_argument('--path_test', help='Specify file path to the test set.', type=str, required=True)
parser.add_argument('--model_name', help='Specify the model name', type=str, required=True)
parser.add_argument('--target', help='Supported target variables ["sic2", "sic4"].', type=str, default="sic2")
parser.add_argument('--custom_path', help='path to custom model', type=str, default="")
parser.add_argument('--wandb_project', help='model name', type=str, default="IC-SIC2")
parser.add_argument('--wandb_dir', help='model name', type=str, required="./wandb")
parser.add_argument('--output_dir', help='output directory', type=str, default="./runs")
parser.add_argument('--max_seq_len', help='output directory', type=int, default=512)
parser.add_argument('--seed', help='random seed', type=int, default=42)
parser.add_argument('--per_device_train_batch_size', help='batch size', type=int, default=16)
parser.add_argument('--gradient_accumulation_steps', help='gradient accumulation steps', type=int, default=2)
parser.add_argument('--learning_rate', help='learning rate', type=float, default=5e-5)
parser.add_argument('--num_train_epochs', help='nr training epochs', type=int, default=10)
args=parser.parse_args()

wandb.login()

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def read_data(path_train, path_valid, path_test):
    # read data
    train = pd.read_parquet(path_train)
    valid = pd.read_parquet(path_valid)
    test = pd.read_parquet(path_test)
    num_labels = len(np.unique(train[args.target]))
    return train,valid,test,num_labels

  
# Define pretrained tokenizer and model
def init_model(model_name, num_labels):
    if model_name.startswith("bert") or model_name.startswith("Prosus"):
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    elif model_name.startswith("roberta"):
        tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_name.startswith("custom"):
        model_path = f"{args.custom_path}/pytorch_model.bin"
        vocab_path = f"{args.custom_path}/vocab.txt"
        config_path = f"{args.custom_path}/config.json"
        tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
        model = BertForSequenceClassification.from_pretrained(model_path, config=config_path, num_labels=num_labels)
    
    return model, tokenizer

  
# Dataset class
class ECData(Dataset):
    def __init__(self, data, tokenizer, max_length=512, **kwargs):
        super().__init__()

        self.X = np.asarray(data["text"])
        self.y = np.asarray(data[args.target].astype(int))
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):

        # get batch
        X = self.X[index]
        label = self.y[index]
        
        # process text
        encoded = self.tokenizer.encode_plus(X, return_attention_mask=True, return_token_type_ids=False,padding='max_length', max_length=self.max_length, truncation=True, add_special_tokens=True, return_tensors='pt')
        for key,val in encoded.items(): encoded[key] = val.squeeze(0) 
        encoded["labels"] = torch.as_tensor(label, dtype=torch.int64)
        
        return encoded

  
# get data loaders
def get_dl(train, valid, test, tokenizer, max_length, **kwargs):
    data_train  = ECData(data=train, tokenizer=tokenizer, max_length=max_length,**kwargs)
    dl_train    = DataLoader(dataset=data_train, batch_size=kwargs.get("batch_size", 16), shuffle=True)

    data_valid  = ECData(data=valid, tokenizer=tokenizer, max_length=max_length,**kwargs)
    dl_valid    = DataLoader(dataset=data_valid, batch_size=kwargs.get("batch_size", 16), shuffle=False)

    data_test  = ECData(data=test, tokenizer=tokenizer, max_length=max_length,**kwargs)
    dl_test    = DataLoader(dataset=data_test, batch_size=kwargs.get("batch_size", 16), shuffle=False)

    # dl = {'train':dl_train, 'valid':dl_valid, 'test':dl_test}
    dl = {'train':data_train, 'valid':data_valid, 'test':data_test}

    return dl

# evaluation metrics
def eval_metrics(eval_preds):
    logits, labels = eval_preds
    pred = np.argmax(logits, axis=-1)

    # print(np.bincount(pred))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy             = accuracy_score(labels, pred)
        f1_weighted          = f1_score(labels, pred, average="weighted")
        precision_weighted   = precision_score(labels, pred, average="weighted")
        recall_weighted      = recall_score(labels, pred, average="weighted")
        f1_micro             = f1_score(labels, pred, average="micro")
        precision_micro      = precision_score(labels, pred, average="micro")
        recall_micro         = recall_score(labels, pred, average="micro")

    res_dict = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
    }

    return res_dict


# Evaluate
def evaluate(trainer, data):
    pred_test = trainer.predict(data)
    wandb.log({"test":wandb.Table(data=list(pred_test.metrics.items()),columns = ["metric", "value"])})
    print(pred_test.metrics)

def main():
    model_name = args.model_name

    # get data
    X_train, X_valid, X_test, num_labels = read_data(args.path_train, args.path_valid, args.path_test)
    model, tokenizer = init_model(model_name, num_labels)

    # get dataloaders
    dl = get_dl(X_train, X_valid, X_test, tokenizer=tokenizer, max_length=args.max_seq_len, batch_size=args.per_device_train_batch_size, model_name=model_name)

    # name for custom models
    if model_name == "custom":
        model_name = args.custom_path.split("/")[-1]

    elif model_name == "ProsusAI/finbert":
        model_name = "ProsusAI"

    wandb.init(
        project=args.wandb_project,
        name=f"{model_name}",
        dir=args.wandb_dir,
        reinit=True,
    )

    # setup trainer
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{model_name}_{args.target}",
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="epoch", 
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="wandb",
        run_name=f"{model_name}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dl["train"],
        eval_dataset=dl["valid"],
        compute_metrics=eval_metrics,
    )

    # train
    trainer.train()

    # eval
    evaluate(trainer, dl["test"])
    wandb.finish()

if __name__ == "__main__":
    main()
