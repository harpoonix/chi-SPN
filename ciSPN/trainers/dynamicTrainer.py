import math

import torch
from rtpt import RTPT
from ciSPN.environment import environment
from torch.optim.lr_scheduler import MultiStepLR, SequentialLR

torch.autograd.set_detect_anomaly(True)
class DynamicTrainer:
    def __init__(self, model, conf, loss, train_loss=False, lr=1e-3, pre_epoch_callback=None, optimizer="adam", scheduler_fn=None):
        self.model = model
        self.conf = conf
        self.base_lr = lr

        self.loss = loss

        trainable_params = [{'params': model.parameters(), 'lr': lr}]
        if train_loss:
            loss.train()
            trainable_params.append({'params': loss.parameters(), 'lr': lr})
        else:
            loss.eval()

        print(f"Optimizer: {optimizer}")
        print("Learning Rate: {}".format(lr))

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(trainable_params) #, amsgrad=True)
            self.scheduler = MultiStepLR(self.optimizer, milestones=[4, 7, 10, 14, 20], gamma=0.2)        
        elif optimizer == "adamWD":
            self.optimizer = torch.optim.Adam(trainable_params, weight_decay=0.01)
            self.scheduler = None
        elif optimizer == "adamAMS":
            self.optimizer = torch.optim.Adam(trainable_params, amsgrad=True)
            self.scheduler = None
        elif optimizer == "adamW":
            self.optimizer = torch.optim.AdamW(trainable_params) #, amsgrad=True)
            self.scheduler = None
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(trainable_params, lr, momentum=0.9)
            self.scheduler = scheduler_fn(self.optimizer) if scheduler_fn is not None else None
        elif optimizer == "sgdWD":
            self.optimizer = torch.optim.SGD(trainable_params, lr, momentum=0.9, weight_decay=0.01)
            self.scheduler = scheduler_fn(self.optimizer) if scheduler_fn is not None else None
        else:
            raise ValueError(f"unknown optimizer {optimizer}")

        self.pre_epoch_callback = pre_epoch_callback
        

    def run_training(self, provider):
        nll_loss_curve = []
        cfd_loss_curve = []
        loss_value = -math.inf

        # set spn into training mode
        self.model.cuda().train()

        loss_name = self.loss.get_name()

        rtpt = RTPT(name_initials=environment["runtime"]["initials"], experiment_name='CausalLoss Training', max_iterations=self.conf.num_epochs)
        rtpt.start()

        for epoch in range(self.conf.num_epochs):
            if self.pre_epoch_callback is not None:
                self.pre_epoch_callback(epoch, loss_value, self.loss)

            batch_num = 0
            while provider.has_data():
                x, y = provider.get_next_batch()

                self.optimizer.zero_grad(set_to_none=True)
                self.loss.zero_grad(set_to_none=True)

                cur_loss = self.model.forward(x, y)

                cur_loss.backward()
                max_grad = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.abs().max().cpu().item())
                        
                if (not max_grad < 2):
                    print(f'max_grad: {max_grad}')
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                # print(f'done')
                
                if batch_num % 10 == 0:
                    nll_loss = self.loss.forward(x, y, self.model.logpdf(x,y))
                    nll_loss_np = nll_loss.cpu().item()
                    cur_loss_np = cur_loss.cpu().item()
                    print(f'ep. {epoch}, batch {batch_num}, train {loss_name} {nll_loss_np:.2f} cfd lossx1000 {1000*cur_loss_np}', flush=True)
                batch_num += 1

            rtpt.step(f"ep{epoch}/{self.conf.num_epochs}")

            nll_loss_value = nll_loss.detach().cpu().item()
            cfd_loss_value = 1000*cur_loss.detach().cpu().item()
            nll_loss_curve.append(nll_loss_value)
            cfd_loss_curve.append(cfd_loss_value)

            provider.reset()

            if self.scheduler is not None:
                self.scheduler.step()

        return zip(nll_loss_curve, cfd_loss_curve)

    def run_training_dataloader(self, dataloader, batch_processor):
        loss_curve = []
        loss_value = -math.inf

        # set spn into training mode
        self.model.cuda().train()

        loss_name = self.loss.get_name()

        rtpt = RTPT(name_initials=environment["runtime"]["initials"], experiment_name='CausalLoss Training', max_iterations=self.conf.num_epochs)
        rtpt.start()

        for epoch in range(self.conf.num_epochs):
            if self.pre_epoch_callback is not None:
                self.pre_epoch_callback(epoch, loss_value, self.loss)

            for batch_num, batch in enumerate(dataloader):
                x, y = batch_processor(batch)

                self.optimizer.zero_grad(set_to_none=True)
                self.loss.zero_grad(set_to_none=True)

                prediction = self.model.forward(x, y)
                cur_loss = self.loss.forward(x, y, prediction)

                cur_loss.backward()
                self.optimizer.step()

                if batch_num % 100 == 0:
                    cur_loss_np = cur_loss.cpu().item()
                    print(f'ep. {epoch}, batch {batch_num}, train {loss_name} {cur_loss_np:.2f}', end='\r', flush=True)

            rtpt.step(f"ep{epoch}/{self.conf.num_epochs}")

            loss_value = cur_loss.detach().cpu().item()
            loss_curve.append(loss_value)

            if self.scheduler is not None:
                self.scheduler.step()

        return loss_curve
