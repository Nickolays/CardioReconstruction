import torch


isCuda = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and isCuda) else "cpu")
# for ep in range(start_epoch, epochs):
# update_lr(ep, 4)
# timestamp = time()
# Before train step
def train_step(model, optimizer, dataloader, loss_func):
    step, loss_sum = 0, 0.
    
    for batch_idx, (input, target) in enumerate(dataloader, 0):
        input = input.to(device)
        target = target.to(device)
        step += 1
        
        # Train Model
        model.train()
        optimizer.zero_grad()
        
        out = model(input)
        # each_loss = [BCE(o.unsqueeze(1), target) for o in out]
        # loss = sum(each_loss)
        each_loss = [ loss_func(o, target) for o in out ]  
        loss = sum(each_loss)
        
        loss.backward()
        optimizer.step()

        return loss
# After train step
# Append Loss
# losses['train'].append(loss.item())
# loss_sum += loss.item()

# if (batch_idx+1) % n_print == 0 or batch_idx == (len(dataloader)-1):
#     print('[%2d/%2d][%4d/%4d] Train: %.4f (%ds)' % (ep+1, epochs, batch_idx+1, len(dataloader), loss_sum/step, time() - timestamp))
#     step, loss_sum = 0, 0.

# timestamp = time()