import numpy as np
import copy
import time
# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# TQDM
import tqdm as tqdm
from tqdm import trange

from src.utils import io as utils_io
from src.methods.coral import coral_loss, SimpleFeatureNet
from src.methods.dann import DANNModel, FeatureExtractor, LabelClassifier, DomainClassifier

    
#############################################
###### Reusable Training & Eval Loops
##############################################

def standard_training_epoch(model, optimizer, criterion, train_dataloader, device, args):
    """
    Performs one epoch of standard training.
    This was the original 'training_loop' function.
    """
    train_loss = 0
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        outputs = outputs[1] if isinstance(outputs, tuple) else outputs
        
        # Calculate loss
        loss = criterion(outputs, batch_y)
        train_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return train_loss

def dann_training_epoch(model, optimizer, criterion, train_dataloader, device, args, **kwargs):
    
    """Performs one epoch of training using the DANN loss."""
    target_loaders = kwargs.get('target_loaders')
    if not target_loaders:
        raise ValueError("dann_training_epoch requires 'target_loaders' in kwargs.")
    
    criterion_task = criterion
    criterion_dom = nn.CrossEntropyLoss()
    
    # Unpack the model components from the container
    feat, clf, dom = model.feature_extractor, model.label_classifier, model.domain_classifier
    
    total_task_loss = 0
    total_dom_loss = 0
    
    for batch_tuple in zip(train_dataloader, *target_loaders):
        (source_x, source_y) = batch_tuple[0]
        source_x, source_y = source_x.to(device), source_y.to(device)
        
        # 1. Task loss on source data
        source_features = feat(source_x)
        source_logits = clf(source_features)
        loss_task = criterion_task(source_logits, source_y)

        # 2. Domain loss on source and target data
        target_features_list = [feat(b[0].to(device)) for b in batch_tuple[1:]]
        target_features = torch.cat(target_features_list, dim=0)

        domain_features = torch.cat([source_features, target_features], dim=0)
        domain_labels = torch.cat([
            torch.zeros(source_features.size(0), device=device),
            torch.ones(target_features.size(0), device=device)
        ]).long()
        
        domain_logits = dom(domain_features, lambd=1.0) # Pass lambda for GRL
        loss_dom = criterion_dom(domain_logits, domain_labels)
        
        # 3. Combine losses and backpropagate
        loss = loss_task + args.lambda_domain * loss_dom # lambda is applied inside GRL
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_task_loss += loss_task.item()
        total_dom_loss += loss_dom.item()

    # Return task loss for consistent logging with validation loss
    return total_task_loss

def coral_training_epoch(model, optimizer, criterion, train_dataloader, device, args, **kwargs):
    """
    Performs one epoch of training using the CORAL loss for domain adaptation.
    """
    # The main criterion is for the task loss
    criterion_task = criterion
    criterion_coral = coral_loss
    
    # Retrieve the target_loader from kwargs
    target_loader = kwargs.get('target_loader')
    if not target_loader:
        raise ValueError("coral_training_epoch requires a 'target_loader' in kwargs.")

    total_task_loss = 0
    total_coral_loss = 0
    
    # Ensure the loop stops when the shorter dataloader is exhausted
    for (source_x, source_y), (target_x, _) in zip(train_dataloader, target_loader):
        source_x, source_y = source_x.to(device), source_y.to(device)
        target_x = target_x.to(device)

        # Forward pass for both source and target
        src_feat, src_logits = model(source_x)
        target_feat, _ = model(target_x)

        # Calculate losses
        loss_task = criterion_task(src_logits, source_y)
        loss_coral = criterion_coral(src_feat, target_feat)

        # Combine losses with the lambda hyperparameter
        loss = loss_task + args.lambda_coral * loss_coral

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_task_loss += loss_task.item()
        total_coral_loss += loss_coral.item()

    # We return the task loss to be consistent with validation loss metric
    return total_task_loss

def fixmatch_training_epoch(model, optimizer, criterion, train_dataloader, device, args, **kwargs):
    """
    One epoch of domain adaptation using FixMatch:
    - Labeled source data: supervised loss
    - Labeled target data (partial): supervised loss
    - Unlabeled target data: FixMatch pseudo-label consistency loss

    Required kwargs:
        - target_labeled_loader: dataloader for partially labeled target samples
        - target_unlabeled_loader: dataloader for unlabeled target samples
        - weak_augment, strong_augment: augmentation functions
    """
    model.train()
    total_loss, total_src, total_tgt_lab, total_unsup = 0.0, 0.0, 0.0, 0.0

    # Extract loaders and augmentation functions
    target_labeled_train_loader = kwargs.get('target_labeled_train_loader', None)
    target_unlabeled_loader = kwargs.get('target_unlabeled_loader', None)
    weak_augment = kwargs.get('weak_augment')
    strong_augment = kwargs.get('strong_augment')

    if target_labeled_train_loader is None or target_unlabeled_loader is None:
        raise ValueError("FixMatch requires 'target_labeled_loader' and 'target_unlabeled_loader'.")
    
    # Zip loaders together (min-length iteration)
    for (src_x, src_y), (tgt_x_l, tgt_y_l), (tgt_x_u, _) in zip(train_dataloader, target_labeled_train_loader, target_unlabeled_loader):
        src_x, src_y = src_x.to(device), src_y.to(device)
        tgt_x_l, tgt_y_l = tgt_x_l.to(device), tgt_y_l.to(device)
        tgt_x_u = tgt_x_u.to(device)

        # ============================
        # (1) Supervised loss - Source
        # ============================
        out_src = model(src_x)
        out_src = out_src[1] if isinstance(out_src, tuple) else out_src
        loss_src = criterion(out_src, src_y)

        # =====================================
        # (2) Supervised loss - Target (partial)
        # =====================================
        out_tgt_l = model(tgt_x_l)
        out_tgt_l = out_tgt_l[1] if isinstance(out_tgt_l, tuple) else out_tgt_l
        loss_tgt_lab = criterion(out_tgt_l, tgt_y_l)

        # ===========================================
        # (3) Unsupervised FixMatch loss (target unlabeled)
        # ===========================================
        # weakly and strongly augmented views
        weak_x = weak_augment(tgt_x_u)
        strong_x = strong_augment(tgt_x_u)

        # forward weakly augmented
        with torch.no_grad():
            logits_weak = model(weak_x)
            logits_weak = logits_weak[1] if isinstance(logits_weak, tuple) else logits_weak
            probs_weak = torch.sigmoid(logits_weak)

            # confidence mask for multi-label
            mask = (probs_weak > args.fixmatch_threshold).float()
            pseudo_labels = (probs_weak > 0.5).float()

        # forward strongly augmented
        logits_strong = model(strong_x)
        logits_strong = logits_strong[1] if isinstance(logits_strong, tuple) else logits_strong
        loss_unsup = (F.binary_cross_entropy_with_logits(logits_strong, pseudo_labels, reduction='none') * mask).mean()

        # =======================================
        # Combine all losses with weighting
        # =======================================
        loss = loss_src + args.lambda_target * loss_tgt_lab + args.lambda_unsup * loss_unsup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_src += loss_src.item()
        total_tgt_lab += loss_tgt_lab.item()
        total_unsup += loss_unsup.item()

    return total_loss / len(train_dataloader)

#############################################
###### Reusable Eval Loop
##############################################

def multi_label_eval_loop(model, criterion, dataloader, device, threshold=0.5):

    loss = 0.0
    true_labels, predicted_labels, scores = [], [], []

    model.eval()
    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)        # shape: (B, C)
            outputs = model(inputs)

            # Some models return (features, logits)
            logits = outputs[1] if isinstance(outputs, tuple) else outputs

            batch_loss = criterion(logits, labels)
            loss += batch_loss.item()

            # Probabilities for each class (multi-label)
            prob = torch.sigmoid(logits)      # shape: (B, C)

            # Predictions by thresholding
            preds = (prob >= threshold).int() # shape: (B, C)

            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(preds.cpu().numpy())
            scores.append(prob.cpu().numpy())

    # Average loss
    loss /= len(dataloader)

    # Stack all results
    true_labels = np.vstack(true_labels)
    predicted_labels = np.vstack(predicted_labels)
    scores = np.vstack(scores)

    return loss, true_labels, predicted_labels, scores


#############################################
###### GENERALIZED Training Runner
##############################################

def train_run(model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader,
              args, device, training_epoch_fn, threshold=0.5, **kwargs):
    """
    A generalized training run that accepts a custom function for the training epoch logic.

    Arguments
    ----------------
    ...
    training_epoch_fn: function
        A function that executes one training epoch. It should accept (model, optimizer,
        criterion, train_dataloader, device, args, **kwargs) and return the total training loss for the epoch.
    **kwargs:
        Additional keyword arguments to be passed to the training_epoch_fn (e.g., target_loader).
    """
    target_labeled_val_loader = kwargs.get('target_labeled_val_loader', None)
    target_labeled_test_loader = kwargs.get('target_labeled_test_loader', None)

    epochs = args.num_epochs
    best_val_loss = float('inf')
    best_model_state_dict = None
    best_test_outputs = {}
    return_dict = {}
    training_stats = []
    
    start_time = time.time()
    progress_bar = trange(epochs, desc="Initializing...")

    for epoch in progress_bar:
        model.train()
        t0 = time.time()

        # Call the provided training function for one epoch
        total_train_loss = training_epoch_fn(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            device=device,
            args=args,
            **kwargs
        )

        t1 = time.time()
        model.eval()

        val_loss, val_true_classes, val_predictions, val_scores = multi_label_eval_loop(model, criterion, val_dataloader, device, threshold=threshold)
        if target_labeled_val_loader is not None:
            target_val_loss, target_val_true_classes, target_val_predictions, target_val_scores = multi_label_eval_loop(model, criterion, target_labeled_val_loader, device, threshold=threshold)
        else:
            target_val_loss = None

        
        t2 = time.time()

        if val_loss < best_val_loss:

            return_dict.update({
                'val_true_classes': val_true_classes,
                'val_predictions': val_predictions,
                'val_scores': val_scores})
            
            if target_labeled_val_loader is not None:
                return_dict.update({
                'target_val_true_classes': target_val_true_classes,
                'target_val_predictions': target_val_predictions,
                'target_val_scores': target_val_scores})

            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

            test_loss, test_true_classes, test_predictions, test_scores = multi_label_eval_loop(model, criterion, test_dataloader, device, threshold=threshold)

            best_test_outputs = {
                'test_loss': test_loss,
                'test_true_classes': test_true_classes,
                'test_predictions': test_predictions,
                'test_scores': test_scores
            }

            best_target_test_outputs = {}
            if target_labeled_test_loader is not None:
                print('computing target test')
                tgt_test_loss, tgt_true_classes, tgt_predictions, tgt_scores = multi_label_eval_loop(
                    model, criterion, target_labeled_test_loader, device, threshold=threshold
                )

                best_target_test_outputs = {
                    'target_test_loss': tgt_test_loss,
                    'target_test_true_classes': tgt_true_classes,
                    'target_test_predictions': tgt_predictions,
                    'target_test_scores': tgt_scores
                }

        avg_train_loss = total_train_loss / len(train_dataloader)
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
        
        training_stats.append({
            "epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": val_loss,
            "Target Validation Loss": target_val_loss,
            "Training Time": utils_io.format_time(t1 - t0),
            "Validation Time": utils_io.format_time(t2 - t1),
        })

    print(f'Total training time: {utils_io.format_time(time.time() - start_time)}')

    # Restore best model state
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
    
    return_dict.update({
        'model': model,
        'training_stats': training_stats,
        **best_test_outputs,
        **best_target_test_outputs
    })
    
    return return_dict


#############################################
###### Simplified Trainer Functions
##############################################


def train_coral(train_loader, val_loader, test_loader, target_loader, args, device, threshold=0.5):
    """
    Sets up and runs the training for a domain adaptation model using CORAL.
    This function now uses the generic train_run.
    """
    # 1. Initialize Model, Optimizer, and Criterion
    model = SimpleFeatureNet(args.input_dim, args.feat_dim, args.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_task = nn.BCEWithLogitsLoss()


    # 2. Call the generic training run with the CORAL-specific epoch function
    training_results = train_run(
        model=model,
        optimizer=optimizer,
        criterion=criterion_task,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        threshold=threshold,
        args=args,
        device=device,
        training_epoch_fn=coral_training_epoch,  
        target_loader=target_loader              
    )

    return training_results

def train_dann(train_loader, val_loader, test_loader, target_loader, args, device):

    """Sets up and runs training for a DANN model."""
    # 1. Initialize the three separate networks
    feat = FeatureExtractor(in_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    clf = LabelClassifier(in_dim=args.hidden_dim, n_classes=args.n_classes).to(device)
    dom = DomainClassifier(in_dim=args.hidden_dim).to(device)
    
    # 2. Wrap them in the DANNModel container
    dann_model = DANNModel(feat, clf, dom).to(device)
    
    # 3. Create optimizer over all parameters in the container
    optimizer = torch.optim.Adam(dann_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion_task = nn.BCEWithLogitsLoss()

    # 4. Call the generic training run with the DANN-specific epoch function
    return train_run(
        model=dann_model,
        optimizer=optimizer,
        criterion=criterion_task,
        train_dataloader=train_loader,  
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        args=args,
        device=device,
        training_epoch_fn=dann_training_epoch, # <-- Pass the DANN function
        target_loaders=target_loader              # <-- Pass the extra dataloaders
    )

def train_fixmatch(train_loader, val_loader, test_loader,
                   target_labeled_train_loader, target_labeled_val_loader, target_labeled_test_loader,
                    target_unlabeled_loader, args, device, weak_augment, strong_augment, threshold=0.5):
    """
    Train FixMatch-style domain adaptation with:
      - Source supervised loss
      - Target (partial) supervised loss
      - Unlabeled target FixMatch loss
    """
    # Model setup
    model = SimpleFeatureNet(args.input_dim, args.feat_dim, args.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Use the generic training run with FixMatch-specific epoch
    training_results = train_run(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,     # source labeled
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        args=args,
        device=device,
        threshold=threshold,
        training_epoch_fn=fixmatch_training_epoch,
        target_labeled_train_loader=target_labeled_train_loader,
        target_labeled_val_loader=target_labeled_val_loader,
        target_labeled_test_loader=target_labeled_test_loader,
        target_unlabeled_loader=target_unlabeled_loader,
        weak_augment=weak_augment,
        strong_augment=strong_augment
    )

    return training_results
