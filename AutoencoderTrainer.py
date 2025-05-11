import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


from CubeDataset import get_train_loaders

from utils import get_logger, load_checkpoint, create_optimizer, save_checkpoint, RunningAverage
from utils import _split_and_move_to_gpu, TensorboardFormatter
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import torch

logger = get_logger('AutoencoderTrainer')

class AutoencoderTrainer:
    """UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive)
        resume (string): path to the checkpoint to be resumed
        pre_trained (string): path to the pre-trained model
        max_val_images (int): maximum number of images to log during validation
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations, validate_after_iters=200, log_after_iters=100, validate_iters=None,
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=False, tensorboard_formatter=None,
                 skip_train_validation=False, resume=None, pre_trained=None, max_val_images=100, **kwargs):

        self.max_val_images = max_val_images
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders

        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                checkpoint_dir, 'logs',
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.model, None)
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()

        # sets the model in training mode
        self.model.train()

        for t in self.loaders['train']:
            if self.num_iterations % 1000 == 0:
              logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. 'f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')
              # log the both loss
              if hasattr(self.loss_criterion, "debug_log"):
                  self.loss_criterion.debug_log(self.num_iterations)


            input, target = _split_and_move_to_gpu(t)

            output, loss, encoder_feats = self._forward_pass(input, target)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                eval_score = self.validate()
                # set the model back to training mode
                self.model.train()

                # adjust learning rate if necessary
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(eval_score)
                elif self.scheduler is not None:
                    self.scheduler.step()

                # log current learning rate in tensorboard
                self._log_lr()
                # remember the best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output, target)
                    train_eval_scores.update(eval_score.item(), self._batch_size(input))

                logger.info(
                    f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                self._log_images(
                    input.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                    output.detach().cpu().numpy(),
                    'train_',
                    [f.detach().cpu().numpy() for f in encoder_feats],
                    log_target=False
                )

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()

        with torch.no_grad():
            # select indices of validation samples to log
            rs = np.random.RandomState(42)
            if len(self.loaders['val']) <= self.max_val_images:
                indices = list(range(len(self.loaders['val'])))
            else:
                indices = rs.choice(len(self.loaders['val']), size=self.max_val_images, replace=False)

            images_for_logging = []
            for i, t in enumerate(tqdm(self.loaders['val'])):
                input, target = _split_and_move_to_gpu(t)

                output, loss, encoder_feats = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))
                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                # save val images for logging
                if i in indices:
                    imgs = (
                        input.cpu().numpy(),
                        target.cpu().numpy(),  # 雖然不記錄 target，但保留以便萬一有 log_target=True
                        output.cpu().numpy(),
                        [f.cpu().numpy() for f in encoder_feats]  # encoder features
                    )
                    images_for_logging.append(imgs + (i,))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            # log images in a separate thread
            with ThreadPoolExecutor() as executor:
                for input, target, output, encoder_feats, i in images_for_logging:
                    executor.submit(
                        self._log_images,
                        input,
                        target,
                        output,
                        f'val_{i}_',
                        encoder_feats=encoder_feats,
                        log_target=False  # <<< 不紀錄 target
                    )

            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            self._log_stats('val', val_losses.avg, val_scores.avg)
            return val_scores.avg

    def _forward_pass(self, inp: torch.Tensor, target: torch.Tensor):
        if isinstance(inp, (tuple, list)) and len(inp) == 4:
            input_std, target_std, input_raw, target_raw = inp
        else:
            input_std, target_std = inp, target
            input_raw, target_raw = None, None

        # forward pass：取得 output、logits、encoder_feats
        output, logits, encoder_feats = self.model(input_std, return_logits=True, return_encoder_feats=True)

        # loss 計算：如果是 L1MSSSIMLoss，則同時傳入 raw 資料
        if isinstance(self.loss_criterion, nn.Module) and self.loss_criterion.__class__.__name__ == "L1MSSSIMLoss":
            loss = self.loss_criterion(logits, target_std, input_raw=output, target_raw=target_raw)
        else:
            loss = self.loss_criterion(logits, target_std)

        return output, loss, encoder_feats

    def _is_best_eval_score(self, eval_score: float) -> bool:
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best: bool):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase: str, loss_avg: float, eval_score_avg: float):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(
            self,
            input: np.ndarray,
            target: np.ndarray,
            prediction: np.ndarray,
            prefix: str,
            encoder_feats: list[np.ndarray] = None,  # 新增
            log_target: bool = False
    ):
        inputs_map = {
            'inputs': input,
            # 'targets': target,
            'predictions': prediction
        }
        if log_target:
            inputs_map['targets'] = target

        if encoder_feats is not None:
            inputs_map['encoder_feat_last3'] = encoder_feats[0]  # 倒數第二層
            inputs_map['encoder_feat_last2'] = encoder_feats[1]  # 最後一層
            inputs_map['encoder_feat_last1'] = encoder_feats[2]

        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, (list, tuple)):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b
            else:
                img_sources[name] = batch

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    @staticmethod
    def _batch_size(input: torch.Tensor) -> int:
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
