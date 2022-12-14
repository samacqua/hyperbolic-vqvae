import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import logging
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, metrics=None,
                 valid_metric_ftns=None):
        super().__init__(model, criterion, optimizer, config, metrics)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.len_val = len(valid_data_loader) if valid_data_loader else 0
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.valid_metric_ftns = valid_metric_ftns or []    # Metric to only calculate during validation.
        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in (self.metric_ftns + self.valid_metric_ftns)], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        ### VQ-VAE train ###

        # train(epoch, self.model, self.data_loader, self.optimizer, False, self.log_step, None, self.writer)

        ### hVAE train ###
        self.model.train()
        self.train_metrics.reset()
        n_steps_logged = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            recon_img, *aux_model_outputs = self.model(data)

            loss, *aux_loss = self.criterion(data, recon_img, target, *aux_model_outputs)
            loss.backward()

            # optional gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)

            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(loss, data, target, recon_img, aux_model_outputs, aux_loss))

            if batch_idx % self.log_step == 0:
                n_steps_logged += 1

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

                if data.shape == recon_img.shape:
                    self.writer.add_image('input', make_grid(
                        data.cpu(), nrow=8, normalize=False))
                    self.writer.add_image('recon', make_grid(
                        recon_img.cpu(), nrow=8, normalize=False))

                # try:
                #     if n_steps_logged == 1:
                #         codebooks = self.model.plot_codebooks()
                #         self.writer.add_image('codebooks', make_grid(
                #             codebooks.cpu(), nrow=int(np.sqrt(self.model.k)), normalize=False))
                #
                #         self.model.visualize_im_codebooks(data[0,:,:,:])
                # except Exception as e:
                #     raise e    # only implemented for VQ-VAE -- very hacky but just testing it out.


            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():

            val_mets = {}

            self.writer.set_step(epoch, 'valid')

            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, *aux_model_outputs = self.model(data)
                loss, *aux_loss = self.criterion(data, output, target, *aux_model_outputs)

                val_mets.setdefault('loss', []).append(loss.item())
                for met in self.metric_ftns + self.valid_metric_ftns:
                    val_mets.setdefault(met.__name__, []).append(
                        met(loss, data, target, output, aux_model_outputs, aux_loss))

                self.logger.debug('Valid Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx, val=True),
                    loss.item()))

            for met_name, met_res in val_mets.items():
                self.valid_metrics.update(met_name, float(sum(met_res) / len(met_res)))

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx, val=False):
        base = '[{}/{} ({:.0f}%)]'
        dl = self.valid_data_loader if val else self.data_loader
        if hasattr(dl, 'n_samples'):
            current = batch_idx * dl.batch_size
            total = dl.n_samples
        else:
            current = batch_idx
            total = self.len_val if val else self.len_epoch
        return base.format(current, total, 100.0 * current / total)


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, writer, dataset='mnist', k=10,
          max_epoch_samples=50000):
    model.train()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    print(model)
    print(sum([np.prod(p.size()) for p in model_parameters]))

    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None

    for batch_idx, (data, _) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()

            for key in latest_losses:
                losses[key + '_train'] = 0

        if batch_idx == (len(train_loader) - 1):
            print("ignoring saving reconstruction.")
            # save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')
            # write_images(data, outputs, writer, 'train')

        if dataset in ['imagenet', 'custom'] and batch_idx * len(data) > max_epoch_samples:
            break

    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)

    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    # writer.add_histogram('dict frequency', outputs[3], bins=range(k + 1))
    # model.print_atom_hist(outputs[3])
    return epoch_losses
