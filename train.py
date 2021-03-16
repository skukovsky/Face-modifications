import torch 
from tqdm import tqdm

def fit_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    processed_data = 0
  
    for inputs in train_loader:
        inputs = inputs.view(-1, 45 * 45 * 3)

        inputs = inputs.to(device).type(torch.float32)

        optimizer.zero_grad()

        reconstruction, latent_code = model(inputs)
        
        outputs = reconstruction.view(-1, 45 * 45 * 3)

        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        processed_data += inputs.size(0)
              
    train_loss = running_loss / processed_data
    return train_loss

def eval_epoch(model, val_loader, X_val, criterion):
    model.eval()
    running_loss = 0.0
    processed_size = 0

    for inputs in val_loader:
        inputs = inputs.view(-1, 45 * 45 * 3)
        inputs = inputs.to(device).type(torch.float32)

        with torch.set_grad_enabled(False):

            reconstruction, latent_code = model(inputs)
            outputs = reconstruction.view(-1, 45 * 45 * 3)
            
                
            loss = criterion(outputs, inputs)

        running_loss += loss.item() * inputs.size(0)
        processed_size += inputs.size(0)
        
    val_loss = running_loss / processed_size
    
    with torch.set_grad_enabled(False):
        pic_inputs = X_val[:6]
        
        pic_inputs = pic_inputs.view(-1, 45 * 45 * 3)
        pic_inputs = torch.FloatTensor(pic_inputs)

        pic_inputs = pic_inputs.to(device).type(torch.float32)

        reconstruction, latent_code = model(pic_inputs)
        
        pic_outputs = reconstruction.view(-1, 45 * 45 * 3).squeeze()
        
        pic_outputs = pic_outputs.to("cpu")        
        pic_inputs = pic_inputs.to("cpu")

        plot_gallery(pic_inputs, pic_outputs, n_samples=6)
        
    

    return val_loss

def train(train_loader, val_loader, X_val, model, opt, criterion, epochs):

    history = []
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \ val_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for epoch in range(epochs):
            train_loss = fit_epoch(model, train_loader, criterion, opt)
            print("loss", train_loss)

            val_loss = eval_epoch(model, val_loader, X_val, criterion)
            history.append((train_loss, val_loss))

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch+1, t_loss=train_loss,\
                                           v_loss=val_loss))
            
    return history