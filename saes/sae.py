from abc import ABC, abstractmethod
import torch 
from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'

class SAETemplate(torch.nn.Module, ABC):
    '''
    abstract base class that defines the SAE contract
    '''

    def __init__(self, gpt:GPTforProbing, window_start_trim:int, window_end_trim:int):
        super().__init__()
        self.gpt=gpt
        for param in self.gpt.parameters():
            #freezes the gpt model  
            param.requires_grad=False 
        self.window_start_trim=window_start_trim
        self.window_end_trim=window_end_trim

    def trim_to_window(self, input, offset=0):
        '''
        trims the tensor input from shape (n_batch, l_window, d_model) to (n_batch, l_window - window_start_trim - window_end_trim, d_model)'''
        window_length=input.shape[1]
        return input[:, (self.window_start_trim+offset):(window_length-self.window_end_trim+offset+1), :]

    def forward_on_tokens(self, token_sequences):
        '''
        runs the SAE on a token sequence

        in particular, takes the intermediate layer of the gpt model on this token sequence, trims it to the right part of the context window, and runs the SAE on that residual stream
        '''
        residual_stream=self.gpt(token_sequences)
        trimmed_residual_stream=self.trim_to_window(residual_stream)
        return self.forward(trimmed_residual_stream)

    def forward_on_tokens_with_loss(self, token_sequences):
        '''
        runs the SAE on a token sequence, also returning the loss
        '''
        residual_stream, hidden_layer, reconstructed_residual_stream=self.forward_on_tokens(token_sequences)
        loss = self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream)
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def catenate_outputs_on_dataset(self, dataset, batch_size=8):
        '''
        runs the model on the entire dataset, one batch at a time, catenating the outputs
        '''
        losses=[]
        residual_streams=[]
        hidden_layers=[]
        reconstructed_residual_streams=[]
        test_dataloader=iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))
        for test_input, test_labels in test_dataloader:
            test_input=test_input.to(device)
            loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.forward_on_tokens_with_loss(test_input)
            losses.append(loss)
            residual_streams.append(residual_stream)
            hidden_layers.append(hidden_layer)
            reconstructed_residual_streams.append(reconstructed_residual_stream)
        losses=torch.stack(losses)
        residual_streams=torch.cat(residual_streams)
        hidden_layers=torch.cat(hidden_layers)
        reconstructed_residual_streams=torch.cat(reconstructed_residual_streams)
        return losses, residual_streams, hidden_layers, reconstructed_residual_streams

    def print_evaluation(self, train_loss, eval_dataset, step_number="N/A"):
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset)
        test_loss=losses.mean()
        l0_sparsity=self.compute_l0_sparsity(hidden_layers)
        dead_features=self.count_dead_features(hidden_layers)
        print_message=f"Train loss, test loss, l0 sparsity, dead features after {step_number} steps: {train_loss.item():.2f}, {test_loss:.2f}, {l0_sparsity:.1f}, {dead_features:.0f}"
        tqdm.write(print_message)

    def compute_l0_sparsity(self, hidden_layers):
        active_features=hidden_layers>0
        sparsity_per_entry=active_features.sum()/hidden_layers[..., 0].numel()
        return sparsity_per_entry

    def count_dead_features(self, hidden_layers):
        active_features=hidden_layers>0
        dead_features=torch.all(torch.flatten(active_features, end_dim=-2), dim=0)
        num_dead_features=dead_features.sum()
        return num_dead_features

    @abstractmethod
    def forward(self, residual_stream):
        '''
        takes the trimmed residual stream of a language model (as produced by run_gpt_and_trim) and runs the SAE
        must return a tuple (residual_stream, hidden_layer, reconstructed_residual_stream)
        residual_stream is shape (B, W, D), where B is batch size, W is (trimmed) window length, and D is the dimension of the model:
            - residual_stream is unchanged, of size (B, W, D)
            - hidden_layer is of shape (B, W, D') where D' is the size of the hidden layer
            - reconstructed_residual_stream is shape (B, W, D) 
        '''
        pass

    @abstractmethod
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        '''
        loss function which depends solely on residual stream, hidden layer, and reconstruction
        '''
        pass


class SAEAnthropic(SAETemplate):

    def __init__(self, gpt:GPTforProbing, feature_ratio:int, sparsity_coefficient:float, window_start_trim:int, window_end_trim:int):
        super().__init__(gpt=gpt, window_start_trim=window_start_trim, window_end_trim=window_end_trim)
        self.feature_ratio=feature_ratio
        self.sparsity_coefficient=sparsity_coefficient
        residual_stream_size=gpt.pos_emb.shape[-1]
        self.encoder=torch.nn.Parameter(torch.rand((residual_stream_size, residual_stream_size*feature_ratio)))
        self.encoder_bias=torch.nn.Parameter(torch.rand((residual_stream_size*feature_ratio)))
        self.decoder=torch.nn.Parameter(torch.rand((residual_stream_size*feature_ratio, residual_stream_size)))
        self.decoder_bias=torch.nn.Parameter(torch.rand((residual_stream_size)))


    def forward(self, residual_stream):
        hidden_layer=torch.nn.functional.relu(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        return residual_stream, hidden_layer, reconstructed_residual_stream
    
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_l2=torch.norm(reconstructed_residual_stream-residual_stream, dim=-1)
        reconstruction_loss=(reconstruction_l2**2).mean()
        sparsity_loss= torch.mean(hidden_layer)*self.sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss

def train_model(model:SAETemplate, train_dataset, eval_dataset, batch_size=64, num_epochs=2, report_every_n_steps=500, fixed_seed=1337):
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
        
        for input_batch, label_batch in tqdm(train_dataloader):
            input_batch=input_batch.to(device)
            step+=1
            optimizer.zero_grad(set_to_none=True)
            loss, residual_stream, hidden_layer, reconstructed_residual_stream= model.forward_on_tokens_with_loss(input_batch)
            loss.backward()
            optimizer.step()
            if step % report_every_n_steps==0:
                model.print_evaluation(loss, eval_dataset, step_number=step)
    else:
        model.print_evaluation(train_loss=loss, eval_dataset=eval_dataset, step_number="Omega")
