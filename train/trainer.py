import logging
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch
import wandb
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from models.model_type import ModelType
from utils.model_utils import count_parameters, count_all_parameters


class Trainer:

    def __init__(self, model_name=None, n_conditional_branches=1, wandb_project=None, save_model=False):
        if torch.cuda.is_available():
            logging.info("Using CUDA")
            self.device = torch.device("cuda")
        else:
            logging.info("Using CPU")
            self.device = torch.device('cpu')
        self.n_conditional_branches = n_conditional_branches
        self.wandb_logging = False
        self.save_model = save_model # TODO replace with callback

        if wandb_project:
            self.wandb_logging = True

        if model_name:
            datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
            self.save_dir = Path("trained_models") / f"{datetime_prefix}_{model_name}"
            self.save_dir.mkdir(parents=True, exist_ok=False)
        self.train_batch_count = None
        self.valid_batch_count = None

    def train(self, model, model_type, train_loader, valid_loader, optimizer, criterion, postprocessors,
              build_evaluators_fn, n_epoch, patience=10, lr_patience=10):

        model = model.to(self.device)
        criterion = criterion.to(self.device)

        #TODO: Init hook
        if model_type == ModelType.PILOTNET:
            # When using LazyModules Call `forward` with a dummy batch to initialize the parameters
            # before calling torch functions
            data, _, _ = next(iter(train_loader))
            inputs = data['image'].to(self.device)
            model(inputs)

        if self.wandb_logging:
            wandb.watch(model, criterion)

        best_valid_loss = float('inf')
        epochs_of_no_improve = 0

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=0.1, verbose=True)

        num_params = count_parameters(model)
        num_params_all = count_all_parameters(model)

        logging.info("Model: %s number of all parameters: %s, trainable parameters: %s", model_type, num_params_all, num_params)

        self.train_batch_count = dataset_len(train_loader)
        self.valid_batch_count = dataset_len(valid_loader)

        for epoch in range(n_epoch):

            progress_bar = tqdm(total=self.train_batch_count, smoothing=0)
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, progress_bar, epoch)

            progress_bar.reset(total=self.valid_batch_count)
            valid_loss, val_stats = self.evaluate(model, valid_loader, criterion, build_evaluators_fn(),
                                                  postprocessors, progress_bar, epoch, train_loss)

            scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(model.state_dict(), self.save_dir / f"best.pt")
                torch.save(model.state_dict(), self.save_dir / f"best-{epoch}.pt")
                epochs_of_no_improve = 0
                best_loss_marker = '*'
            else:
                epochs_of_no_improve += 1
                best_loss_marker = ''

            metrics = val_stats

            log_stats_str = ' | '.join([f'{k}: {v}' for k, v in val_stats.items()])

            progress_bar.set_description(f'{best_loss_marker}epoch {epoch + 1}/{n_epoch}'
                                         f' | train loss: {train_loss:.4f}'
                                         f' | valid loss: {valid_loss:.4f}'
                                         f' | {log_stats_str}'
                                         )

            if self.wandb_logging:
                metrics['epoch'] = epoch + 1
                metrics['train_loss'] = train_loss
                metrics['valid_loss'] = valid_loss
                wandb.log(metrics)

            if epochs_of_no_improve == patience:
                logging.info(f'Early stopping, on epoch: {epoch + 1}.')
                break

        self.save_models(model, valid_loader)

        return best_valid_loss

    def save_models(self, model, valid_loader):
        if not self.save_model:
            return

        torch.save(model.state_dict(), self.save_dir / "last.pt")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.pt")
            wandb.save(f"{self.save_dir}/best.pt")

        self.save_onnx(model, valid_loader)

    def save_onnx(self, model, valid_loader):
        model.load_state_dict(torch.load(f"{self.save_dir}/best.pt"))
        model = model.to(self.device)

        #data = iter(valid_loader).next()
        #Update to fix an issue of deprecated code.
        data = iter(valid_loader)
        data = next(data)
        sample_inputs = self.create_onxx_input(data)
        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/best.onnx")
        onnx.checker.check_model(f"{self.save_dir}/best.onnx")
        #Remove batch size from the input/output
        m = onnx.load(f"{self.save_dir}/best.onnx")
        m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        m.graph.output[0].type.tensor_type.shape.dim[0].dim_value =1
        onnx.save(m,f"{self.save_dir}/best.onnx")

        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/best.onnx")

        model.load_state_dict(torch.load(f"{self.save_dir}/last.pt"))
        model = model.to(self.device)

        torch.onnx.export(model, sample_inputs, f"{self.save_dir}/last.onnx")
        onnx.checker.check_model(f"{self.save_dir}/last.onnx")
        if self.wandb_logging:
            wandb.save(f"{self.save_dir}/last.onnx")

    def create_onxx_input(self, data):
        return data[0]['image'].to(self.device)

    def train_epoch(self, model, loader, optimizer, criterion, progress_bar, epoch):
        running_loss = 0.0
        batch_count = 0

        model.train()
        for i, loader_data in enumerate(loader):
            optimizer.zero_grad()

            predictions, loss = self.train_batch(model, criterion, loader_data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'epoch {epoch+1} | train loss: {(running_loss / (i + 1)):.4f}')
            batch_count += 1
        self.train_batch_count = batch_count
        return running_loss / batch_count

    def train_batch(self, model, criterion, loader_data):
        sample, targets = loader_data
        sample = sample.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        predictions, _ = model(sample)
        loss = criterion(predictions, targets)
        return predictions, loss

    @abstractmethod
    def predict(self, model, dataloader):
        pass

    def evaluate(self, model, iterator, criterion, evaluators, postprocessors, progress_bar, epoch, train_loss):
        epoch_loss = 0.0
        model.eval()
        batch_count = 0
        stats = {}

        with torch.no_grad():
            for i, loader_data in enumerate(iterator):
                predictions, loss = self.train_batch(model, criterion, loader_data)
                epoch_loss += loss.item()

                progress_bar.update(1)
                progress_bar.set_description(f'epoch {epoch + 1} | train loss: {train_loss:.4f} | valid loss: {(epoch_loss / (i + 1)):.4f}')
                batch_count += 1

                for evaluator in evaluators:
                    evaluator.update(postprocessors, loader_data, predictions)

        for evaluator in evaluators:
            evaluator.fill_stats(stats)

        self.valid_batch_count = batch_count
        total_loss = epoch_loss / batch_count
        return total_loss, stats


def dataset_len(dataloader):
    if hasattr(dataloader.dataset, '__len__'):
        return len(dataloader)
    else:
        return None


class PilotNetTrainer(Trainer):

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = data['image'].to(self.device)
                predictions = model(inputs)
                all_predictions.extend(predictions.cpu().squeeze().numpy())
                progress_bar.update(1)

        return np.array(all_predictions)

    def train_batch(self, model, criterion, loader_data):
        data, target_values, condition_mask = loader_data
        inputs = data['image'].to(self.device)
        target_values = target_values.to(self.device)
        predictions = model(inputs)
        return predictions, criterion(predictions, target_values)
    
    
class PerceiverTrainer(Trainer):

    def __init__(self, prepare_dataloader_data_fn, is_many_to_one=False, model_name=None,
                 n_conditional_branches=1, wandb_project=None, save_model=False
                 ):
        super().__init__(model_name, n_conditional_branches, wandb_project, save_model)
        self.prepare_dataloader_data_fn = prepare_dataloader_data_fn
        self.is_many_to_one = is_many_to_one

    def predict(self, model, dataloader):
        all_predictions = []
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(total=len(dataloader), smoothing=0)
            progress_bar.set_description("Model predictions")
            for i, (data, target_values, condition_mask) in enumerate(dataloader):
                inputs = rearrange(data['image'], 'b t c h w -> t b h w c').to(self.device)

                latents = None
                sequence_predictions = []
                
                for t in range(inputs.size(0)):
                    input_frame = inputs[t]

                    predictions, latents = model(input_frame, latents)
                    sequence_predictions.append(predictions)

                all_predictions.append(torch.stack(sequence_predictions).cpu().numpy().squeeze(axis=2))
                progress_bar.update(1)

        return np.concatenate(all_predictions, axis=1)

    def train_batch(self, model, criterion, loader_data):
        model.train()
        inputs, target_values = self.prepare_dataloader_data_fn(loader_data)
        inputs = inputs.to(self.device)
        target_values = target_values.to(self.device)
        latents = None
        sequence_predictions = []
        total_loss = 0.0
        predictions = None
        for t in range(inputs.size(0)):
            input_frame = inputs[t]

            predictions, latents = model(input_frame, latents)

            if not self.is_many_to_one:
                target_frame = target_values[t]
                sequence_predictions.append(predictions)
                # Calculate loss for the current time step
                loss = criterion(predictions.squeeze(), target_frame)
                total_loss += loss

        if self.is_many_to_one:
            sequence_predictions.append(predictions)
            # Calculate loss for the current time step
            loss = criterion(predictions, target_values)
            total_loss += loss

        return torch.stack(sequence_predictions), total_loss
    
    def create_onxx_input(self, data):
        return rearrange(data[0]['image'], 'b t c h w -> t b h w c')[0].to(self.device)
