from .train_utils import *
import os
'''
Full training procedures for a model.

use run_epoch
'''
def train(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, num_epochs, checkpoint_dir, anchor_model=0, task_fn=None, task_model=None, task_optimizer=None, task_epochs=None, binary_only=False):
    # assert mode in ['train', 'val', 'test'], 'mode not in train, val or test'
    assert model_type in ['joint', 'independent', 'sequential'], 'model type not in joint, independent and sequential' 
    '''
    if model_type != 'joint':
        task_fn = kwargs['task_fn']
        task_model = kwargs['task_model']
        task_optim = kwargs['task_optimizer']
    elif model_type == 'independent':
        task_epochs = kwargs['task_epochs']
    '''
    # train independent model
    assert anchor_model in [0,1,2], 'anchor model must be in 0, 1, 2'
    if model_type == 'independent':
        for epoch in range(task_epochs):
            run_task_model_epoch(task_model, model_type, model, device, dataloaders, task_fn, task_optimizer, writer, logger, mode='train', num_epochs=task_epochs, epoch=epoch)
            # val
            if (epoch + 1) % 5 == 0:
                run_task_model_epoch(task_model, model_type, model, device, dataloaders, task_fn, task_optimizer, writer, logger, mode='val', num_epochs=task_epochs, epoch=epoch)
    # train      
    for epoch in range(num_epochs):
        # train joint model
        if model_type == 'joint':
            # train
            run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='train', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
            # val
            if (epoch + 1) % 5 == 0:
                run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='val', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
        else:
            run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='train', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
            # train sequential model
            if model_type == 'sequential':
                # train task model
                run_task_model_epoch(task_model, model_type, model, device, dataloaders, task_fn, task_optimizer, writer, logger, mode='train', num_epochs=task_epochs, epoch=epoch)
            # val
            if (epoch + 1) % 5 == 0:
                run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='val', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
                if model_type == 'independent':
                    task_model.eval()
                run_task_model_epoch(task_model, 'sequential', model, device, dataloaders, task_fn, task_optimizer, writer, logger, mode='val', num_epochs=task_epochs, epoch=epoch, binary_only=binary_only, testing_independent=True)
            
    # test
        if (epoch + 1) % 25 == 0:
            print('test', epoch)
            if model_type == 'joint':
                run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='test', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
                # save models
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{epoch}.pth'))
            elif model_type == 'sequential':
                run_epoch(model, model_type, device, dataloaders, loss_fn, optimizer, attr_group_dict, writer, logger, mode='test', num_epochs=num_epochs, epoch=epoch, anchor_model=anchor_model)
            elif model_type == 'independent':
                run_task_model_epoch(task_model, 'sequential', model, device, dataloaders, task_fn, task_optimizer, writer, logger, mode='test', num_epochs=num_epochs, epoch=epoch, binary_only=binary_only, testing_independent=True)


    

