import torch
from tqdm import tqdm

def av_batch_perplexity(logits):
    """
    For a batch, calculate the average perplexity per-token per-example
    """
    token_probs = logits.softmax(dim=2)  # softmax over the vocab size
    token_perp = token_probs.pow(-token_probs).prod(dim=2)  # get the perplexity per token
    mean_token_perp = token_perp.mean(dim=1)  # get the average perplexity for each example
    return mean_token_perp.mean(dim=0)  # get mean perplexity over the whole batch


def av_batch_accuracy(logits, actual_tokens):
    """
    For a batch, calculate the average accuracy per-token
    """
    most_probable_tokens = logits.softmax(dim=2).argmax(dim=2).squeeze(0)
    pred_equal = most_probable_tokens.eq(actual_tokens).type(torch.uint8)
    per_word_accuracy = pred_equal.sum(dim=1)/pred_equal.shape[1] 
    av_batch_accuracy = per_word_accuracy.mean(dim=0)
    return av_batch_accuracy


def do_train_epoch(model, dataloader, opt):
    """
    Do one epoch of training
    """
    model.train()  # make sure the model is in training mode
    loss = 0
    # store additional metrics
    av_token_perplexity = 0
    av_token_accuracy = 0
    for bn, batch in tqdm(enumerate(dataloader)):
        opt.zero_grad()  # same forward and backward batch size - change this if too expensive
        embeds = model(**batch)
        batch_loss = embeds["loss"] 
        logits = embeds["logits"]
        # do backwards pass
        batch_loss.backward()
        opt.step()
        # update epoch values
        loss += batch_loss
        av_token_perplexity += av_batch_perplexity(logits)
        av_token_accuracy += av_batch_accuracy(logits, batch["labels"])
    # average the metrics over all batches - bn is the number of batches - 1 now
    # so bn + 1 is number of batches
    loss /= bn + 1
    av_token_perplexity /= bn + 1
    av_token_accuracy /= bn + 1
    return {"loss": loss, "perplexity": av_token_perplexity, "accuracy": av_token_accuracy}


def do_val_epoch(model, dataloader):
    """
    Do one validation epoch
    """
    model.eval()  # make sure the model isn't accumulating gradients
    val_loss = 0
    # store additional metrics
    av_token_perplexity = 0
    av_token_accuracy = 0
    for bn, batch in tqdm(enumerate(dataloader)):
        embeds = model(**batch)
        batch_loss = embeds["loss"] 
        logits = embeds["logits"]
        # update epoch values
        val_loss += batch_loss
        av_token_perplexity += av_batch_perplexity(logits)
        av_token_accuracy += av_batch_accuracy(logits, batch["labels"])
    # average the metrics over all batches - bn is the number of batches - 1 now
    # so bn + 1 is number of batches
    val_loss /= bn + 1
    av_token_perplexity /= bn + 1
    av_token_accuracy /= bn + 1
    return {"loss": val_loss, "perplexity": av_token_perplexity, "accuracy": av_token_accuracy}


def summary_string(results_dict, results_type = "train"):
    """
    Parses the results of an epoch into an easily readable string
    """
    assert isinstance(results_type, str), f"results_type must be a str, got {type(results_type)} instead" 
    train_res = [f"{name}: {value}, " for name, value in results_dict.items()]
    summary = "train".join(train_res)
    return summary


def train_model(model, train_loader, optimiser, val_loader = None, n_epochs = 5):
    """
    Train the model on the train dataset, using the validation dataset to validate the model results
    """
    for epoch in range(n_epochs):
        print("="*80)
        train_metrics = do_train_epoch(model, train_loader, optimiser)
        # create the summary string
        summary = f"Epoch {epoch} train " + summary_string(train_metrics)
        print(summary)
        # do validation pass
        if val_loader is not None:
            val_metrics = do_val_epoch(model, val_loader, optimiser)
            # create the summary string
            val_summary = f"Epoch {epoch} val " + summary_string(val_metrics)
            print(val_summary)
        print("="*80)