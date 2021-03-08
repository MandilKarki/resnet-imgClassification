import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy as np
import time
import math
from customDataset import CustomDataset
import hyperparameters as vars
from resnet import *

def set_device():
  global device
  device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
  print(f'Device set to {device}')

def load_data(main_dir, transformations, train_split, batch_size=32,  is_test_set=False, num_workers=0):
    dataset = CustomDataset(main_dir, transform = transformations, is_test_set=is_test_set)
    global num_classes, train_size, valid_size, dataset_size
    num_classes = dataset.get_num_classes()
    dataset_size = len(dataset)
    train_size = math.ceil(train_split*len(dataset))
    valid_size = len(dataset)-train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader    
  
def adam(model, lr):
  return optim.Adam(model.parameters(), lr=lr)

def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

def train_model(train_loader, model, optimizer, loss_func, num_epochs, valid_loader=True):
  model = model_conv
  model.train()
  for epoch in range(1, num_epochs+1):
    start_time = time.time()
    loss_after_epoch, acc_after_epoch = 0, 0
    # images, labels = next(iter(train_loader))
    for images, labels in train_loader:
      torch.cuda.empty_cache()
      images = images
      labels = labels
      preds = model_conv(images)
      loss = loss_func(preds, labels)
      optimizer.zero_grad()
      loss.backward()
      scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
      scheduler.step()
      optimizer.step()

      loss_after_epoch += loss
      acc_after_epoch += get_num_correct(preds, labels)
    
    loss_after_epoch /= 100
    print(f'Epoch:{epoch}/{num_epochs}  Acc:{(acc_after_epoch/train_size):.5f}', end = '  ')
    print(f'Loss:{(loss_after_epoch):.5f}  Duration:{(time.time()-start_time):.2f}s', end='\n' if valid_loader is None else '  ')
    if valid_loader is not None:
      validate_model(model_conv, valid_loader, loss_func)
    if epoch % vars.checkpoint_save_frequency == 0:
      checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'epoch': epoch, 'loss': loss.item()}
      save_checkpoint(checkpoint, vars.checkpoint_dir, vars.checkpoint_filename)

def validate_model(model, loader, loss_func):
  model = model_conv
  model.eval()
  total_val_loss=0
  num_correct = 0
  with torch.no_grad():
    for images,labels in loader:
      images, labels = images, labels
      preds = model_conv(images)
      total_val_loss += loss_func(preds, labels)
      num_correct += get_num_correct(preds, labels)

    total_val_loss = total_val_loss/100
    print(f'Val_acc:{(num_correct/valid_size):.5f}  Val_loss:{(total_val_loss):.5f}')
    scheduler.step((total_val_loss))

def save_checkpoint(state, save_path, filename):
  print('Saving checkpoint...')
  save_path = os.path.join(save_path, filename)
  torch.save(state, save_path)

def get_checkpoint(checkpoint_path):
  print('Loading checkpoint...')
  checkpoint = torch.load(checkpoint_path)
  model_conv.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optimizer'])
  # epoch = checkpoint['epoch']
  # loss = checkpoint['loss']
  return model_conv, optimizer#, epoch, loss

# def early_stopping(measures, delta, patience):
#   pass
#   count=0
#   for measure in measures:
#     measure_plus = measure + delta
#     measure_minus = measure - delta
#     for x in measures:
#       if measure_minus<=x<=measure_plus:
#         count +=1
    
transform_list = [transforms.Resize((224,224)),transforms.ToTensor(),
                  transforms.Normalize(vars.rgb_mean, vars.rgb_std)]


print('Setting device...')
set_device()
print('Creating training and validation dataloaders...')
train_loader, valid_loader = load_data(vars.main_dir, transformations=transforms.Compose(transform_list),
                                       batch_size=vars.batch_size, train_split=vars.train_split, num_workers=vars.num_workers)
print(f'Total of {num_classes} classes were found in the dataset!')
print('Defining model')
print('Initializing the optimizer...')
print('Training the model...')
if vars.load_checkpoint:
  model_conv, optimizer = get_checkpoint(vars.checkpoint_path)
else:
  model = model_conv
  #model= train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=25)
  optimizer= optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

loss_func = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


#trained_model = train_model(train_loader, model, optimizer, loss_func,num_epochs=vars.training_epochs, valid_loader=valid_loader)
model= train_model(train_loader,model_conv, optimizer, loss_func, num_epochs=vars.training_epochs,valid_loader=valid_loader)
