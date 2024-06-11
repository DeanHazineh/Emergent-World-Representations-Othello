import os
import pickle
import torch 
from torcheval.metrics import BinaryAUROC

from EWOthello.data.othello import get
from EWOthello.mingpt.dataset import CharDataset
from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'


def load_pre_trained_gpt(probe_path, probe_layer):
    """
    loads the model at probe_path and wires it to run through probe_layer
    """
    n_layer = int(probe_path[-5:-4])
    n_head = int(probe_path[-3:-2])
    mconf = GPTConfig(61, 59, n_layer=n_layer, n_head=n_head, n_embd=512)
    GPT_probe = GPTforProbing(mconf, probe_layer)
    
    GPT_probe.load_state_dict(torch.load(probe_path + f"GPT_Synthetic_{n_layer}Layers_{n_head}Heads.ckpt", map_location=device))
    GPT_probe.eval()
    return GPT_probe


class SAE(torch.nn.Module):

    def __init__(self, gpt:GPTforProbing, feature_ratio:int, sparsity_coefficient:float, window_start_trim:int, window_end_trim:int):
        super().__init__()
        self.gpt=gpt
        for param in self.gpt.parameters():
            param.requires_grad=False
        self.feature_ratio=feature_ratio
        self.sparsity_coefficient=sparsity_coefficient
        residual_stream_size=gpt.pos_emb.shape[-1]
        self.encoder=torch.nn.Parameter(torch.rand((residual_stream_size, residual_stream_size*feature_ratio)))
        self.encoder_bias=torch.nn.Parameter(torch.rand((residual_stream_size*feature_ratio)))
        self.decoder=torch.nn.Parameter(torch.rand((residual_stream_size*feature_ratio, residual_stream_size)))
        self.decoder_bias=torch.nn.Parameter(torch.rand((residual_stream_size)))
        self.window_start_trim=window_start_trim
        self.window_end_trim=window_end_trim


    def forward(self, input):
        residual_stream=self.gpt(input)
        trimmed_residual_stream=self.trim_to_window(residual_stream)
        hidden_layer=torch.nn.functional.relu(trimmed_residual_stream @ self.encoder + self.encoder_bias)
        reconstruction=hidden_layer @ self.decoder + self.decoder_bias
        return hidden_layer, trimmed_residual_stream, reconstruction
    
    def forward_with_loss(self, input):
        hidden_layer, residual_stream, reconstruction=self.forward(input)
        reconstruction_loss=torch.norm(reconstruction-residual_stream)
        sparsity_loss= torch.mean(hidden_layer)*self.sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss, hidden_layer, residual_stream, reconstruction
    
    def print_evaluation(self, train_loss, eval_dataset, step_number="N/A", details=False):
        del details
        reconstructions, hidden_layers, input_layers, total_losses=self.catenate_outputs_on_test_set(eval_dataset)
        # print(total_losses.mean())
        test_loss=total_losses.mean()
        # percent_active=self.evaluate_sparsity(hidden_layers)
        # percent_dead_neurons=self.evaluate_dead_neurons(hidden_layers)
        # fraction_variance_unexplained=self.evaluate_variance_unexplaiend(reconstructions, input_layers)
        print_message=f"Train loss and test loss after {step_number} steps: {train_loss.item():.4f}, {test_loss:.4f}"
        tqdm.write(print_message)
        # if self.write_updates_to:
        #     with open(self.write_updates_to, 'a') as f:
        #         f.write(print_message + "\n")

    def catenate_outputs_on_test_set(self, eval_dataset):
        test_dataloader=iter(torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False))
        reconstructions=[]
        hidden_layers=[]
        input_layers=[]
        total_losses=[]
        for test_input, test_labels in test_dataloader:
            del test_labels
            test_input=test_input.to(device)

            total_loss, hidden_layer, residual_stream, reconstruction=self.forward_with_loss(test_input)
            reconstructions.append(reconstruction)
            hidden_layers.append(hidden_layer)
            input_layers.append(test_input)
            total_losses.append(total_loss)
        reconstructions=torch.cat(reconstructions)
        hidden_layers=torch.cat(hidden_layers)
        input_layers=torch.cat(input_layers)
        total_losses=torch.stack(total_losses)
        return reconstructions, hidden_layers, input_layers, total_losses

    def trim_to_window(self, x, offset=0):
        window_length=x.shape[1]
        return x[:, (self.window_start_trim+offset):(window_length-self.window_end_trim+offset), :]


def train_model(model, train_dataset, eval_dataset, batch_size=64, num_epochs=2, report_every_n_steps=500, fixed_seed=1337):
    '''
    model be a nn.Module object, and have a print_evaluation() method
    train_dataset and eval_dataset must be in the list of valid types defined in the recognized_dataset() method in utils/dataloaders 
    '''
    if fixed_seed:
        torch.manual_seed(fixed_seed)
    model.to(device)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3)
    step=0
    print(f"Beginning model training on {device}!")


    for epoch in range(num_epochs):
        train_dataloader=iter(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        print(f"Beginning epoch {epoch+1}/{num_epochs}. Epoch duration is {len(train_dataloader)} steps, will evaluate every {report_every_n_steps} steps.")
        
        for input_batch,label_batch in tqdm(train_dataloader):
            input_batch=input_batch.to(device)
            step+=1
            optimizer.zero_grad(set_to_none=True)
            total_loss, hidden_layer, residual_stream, reconstruction=model.forward_with_loss(input_batch)
            total_loss.backward()
            optimizer.step()
            if step %report_every_n_steps==0:
                model.print_evaluation(total_loss, eval_dataset, step_number=step)
    else:
        model.print_evaluation(train_loss=total_loss, eval_dataset=eval_dataset, step_number="Omega", details=True)

# def train(self, game_dataset, num_epochs=1, batch_size=64):
#     optimizer = torch.optim.Adam(self.parameters())
#     train_dataloader = torch.utils.data.DataLoader(game_dataset, batch_size=batch_size, shuffle=True)
#     for epoch in range(num_epochs):
#         for i, (batch_input, batch_labels) in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             total_loss, hidden_layer, residual_stream, reconstruction=self.forward_with_loss(batch_input)
#             total_loss.backward()
#             optimizer.step()
#             print(f"Loss after batch {i}/{len(train_dataloader)}: {float(total_loss)}")


if __name__=="__main__":
    print("Beginning training process. It may take a moment to load the datasets...")
    probe_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
    probe_layer = 6
    GPT_probe=load_pre_trained_gpt(probe_path=probe_path, probe_layer=probe_layer)

    train_dataset = CharDataset(get(ood_num=-1, data_root=None, num_preload=11)) # 11 corresponds to over 1 million games

    test_dataset = CharDataset(get(ood_num=-1, data_root=None, num_preload=1))
    test_set_indices=torch.arange(1000)
    test_1k_dataset = torch.utils.data.Subset(test_dataset, test_set_indices)

    print("\n\n\n")
    print(len(test_dataset))
    print("\n\n\n")

    sae=SAE(gpt=GPT_probe, feature_ratio=2, sparsity_coefficient=.1, window_start_trim=4, window_end_trim=4)
    print("SAE initialized, proceeding to train!")

    train_model(sae, train_dataset, test_1k_dataset)