from abc import ABC, abstractmethod
import torch 
from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
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
        try:
            self.residual_stream_mean=torch.load("saes/model_params/residual_stream_mean.pkl", map_location=device)
            self.average_residual_stream_norm=torch.load("saes/model_params/average_residual_stream_norm.pkl", map_location=device)
        except:
            self.residual_stream_mean=torch.zeros((1))
            self.average_residual_stream_norm=torch.ones((1))
            logger.warning("Please ensure the correct files are in saes/model_params/residual_stream_mean.pkl and saes/model_params/average_residual_stream_norm.pkl!")

    def trim_to_window(self, input, offset=0):
        '''
        trims the tensor input from shape (n_batch, l_window, d_model) to (n_batch, l_window - window_start_trim - window_end_trim, d_model)'''
        window_length=input.shape[1]
        return input[:, (self.window_start_trim+offset):(window_length-self.window_end_trim+offset+1), :]

    def forward_on_tokens(self, token_sequences):
        '''
        runs the SAE on a token sequence

        in particular:
            1. takes the intermediate layer of the gpt model on this token sequence
            2. trims it to the right part of the context window
            3. Normalizes it by subtracting the model mean and dividing by the scale factor
            4. Runs the SAE on that residual stream
        '''
        raw_residual_stream=self.gpt(token_sequences)
        trimmed_residual_stream=self.trim_to_window(raw_residual_stream)
        normalized_residual_stream=(trimmed_residual_stream-self.residual_stream_mean)/self.average_residual_stream_norm
        residual_stream, hidden_layer, reconstructed_residual_stream=self.forward(normalized_residual_stream)
        return residual_stream, hidden_layer, reconstructed_residual_stream

    def forward_on_tokens_with_loss(self, token_sequences):
        '''
        runs the SAE on a token sequence, also returning the loss
        '''
        residual_stream, hidden_layer, reconstructed_residual_stream=self.forward_on_tokens(token_sequences)
        loss = self.loss_function(residual_stream, hidden_layer, reconstructed_residual_stream)
        return loss, residual_stream, hidden_layer, reconstructed_residual_stream

    def catenate_outputs_on_dataset(self, dataset, batch_size=8, include_loss=False):
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
            if include_loss:
                loss, residual_stream, hidden_layer, reconstructed_residual_stream = self.forward_on_tokens_with_loss(test_input)
                losses.append(loss)
            else:
                residual_stream, hidden_layer, reconstructed_residual_stream = self.forward_on_tokens(test_input)
            residual_streams.append(residual_stream)
            hidden_layers.append(hidden_layer)
            reconstructed_residual_streams.append(reconstructed_residual_stream)
        residual_streams=torch.cat(residual_streams)
        hidden_layers=torch.cat(hidden_layers)
        reconstructed_residual_streams=torch.cat(reconstructed_residual_streams)
        if include_loss:
            losses=torch.stack(losses)
            return losses, residual_streams, hidden_layers, reconstructed_residual_streams
        else:
            return residual_streams, hidden_layers, reconstructed_residual_streams

    def print_evaluation(self, train_loss, eval_dataset, step_number="N/A"):
        losses, residual_streams, hidden_layers, reconstructed_residual_streams=self.catenate_outputs_on_dataset(eval_dataset, include_loss=True)
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
        loss function which depends solely on the sae, residual stream, hidden layer, and reconstruction
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
        hidden_layer=self.activation_function(residual_stream @ self.encoder + self.encoder_bias)
        reconstructed_residual_stream=hidden_layer @ self.decoder + self.decoder_bias
        return residual_stream, hidden_layer, reconstructed_residual_stream
    
    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        reconstruction_l2=torch.norm(reconstructed_residual_stream-residual_stream, dim=-1)
        reconstruction_loss=(reconstruction_l2**2).mean()
        sparsity_loss= self.sparsity_loss_function(hidden_layer)*self.sparsity_coefficient
        total_loss=reconstruction_loss+sparsity_loss
        return total_loss

    def activation_function(self, encoder_output):
        return torch.nn.functional.relu(encoder_output)

    def sparsity_loss_function(self, hidden_layer):
        decoder_row_norms=self.decoder.norm(dim=1)
        return torch.mean(hidden_layer*decoder_row_norms)


class SAEDummy(SAETemplate):
    '''
    "SAE" whose hidden layer and reconstruction is just the unchanged residual stream
    '''

    def __init__(self, gpt:GPTforProbing, window_start_trim:int, window_end_trim:int):
        super().__init__(gpt=gpt, window_start_trim=window_start_trim, window_end_trim=window_end_trim)

    def forward(self, residual_stream):
        return residual_stream,residual_stream,residual_stream

    def loss_function(self, residual_stream, hidden_layer, reconstructed_residual_stream):
        return torch.zeros((1))



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

class Smoothed_L0_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, feature_ratio: int, sparsity_coefficient: float, epsilon: float, delta: float, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, feature_ratio, sparsity_coefficient, window_start_trim, window_end_trim)
        self.epsilon = epsilon
        self.delta = delta

    def sparsity_loss_function(self, hidden_layer):
        functions = [CallableConstant(0.0), CallableConstant(1.0)]
        transitions = [{"x":self.epsilon, "epsilon":self.epsilon, "delta":self.delta, "focus":"left"}]
        return torch.mean(smoothed_piecewise(hidden_layer, functions, transitions), dim=-1)
    
class Without_TopK_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, feature_ratio: int, sparsity_coefficient: float, k: int, p: int, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, feature_ratio, sparsity_coefficient, window_start_trim, window_end_trim)
        self.k = k
        self.p = p

    def sparsity_loss_function(self, hidden_layer):
        top_k_indices = torch.topk(torch.abs(hidden_layer), self.k, dim=-1).indices
        top_k_mask = torch.ones(hidden_layer.shape).to(device).scatter_(-1, top_k_indices, 0)
        without_top_k = hidden_layer * top_k_mask
        return torch.mean(torch.norm(without_top_k, p=self.p, dim=-1))

    
class No_Sparsity_Loss_SAE(SAETemplate):
    def __init__(self, gpt: GPTforProbing, feature_ratio: int, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, feature_ratio, 0.0, window_start_trim, window_end_trim)

    def sparsity_loss_function(self, hidden_layer):
        return 0.0
    
class Leaky_Topk_SAE(No_Sparsity_Loss_SAE):
    def __init__(self, gpt: GPTforProbing, feature_ratio: int, epsilon: float, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, feature_ratio, window_start_trim, window_end_trim)
        self.epsilon = epsilon

    def activation_function(self, encoder_output):
        kth_value = torch.topk(torch.abs(encoder_output), k=k).values.min(dim=-1).values
        return suppress_lower_activations(encoder_output, kth_value, epsilon=self.epsilon)

class Dimension_Reduction_SAE(No_Sparsity_Loss_SAE):
    def __init__(self, gpt: GPTforProbing, feature_ratio: int, start_index: int, start_proportion: float, end_proportion: float, epsilon: float, window_start_trim: int, window_end_trim: int):
        super().__init__(gpt, feature_ratio, window_start_trim, window_end_trim)
        self.start_index = start_index
        self.start_proportion = start_proportion
        self.end_proportion = end_proportion
        self.epsilon = epsilon
        self.activation_f = reduce_dimensions_activation(Expand(self.start_index, self.start_proportion, self.end_proportion), max_n = self.hidden_layer_size, epsilon=self.epsilon)

    def activation_function(self, encoder_output):
        return self.activation_f(encoder_output)
    
class CallableConstant(object):
    def __init__(self, constant): self.constant = constant
    def __call__(self, input):
        if torch.is_tensor(input):
            return self.constant * torch.ones(input.shape).to(device)
        else:
            return torch.Tensor((self.constant,)).to(device)

class CallableList(object):
    def __init__(self, list): self.list = torch.Tensor(list).to(device)
    def __call__(self, index):
        if torch.is_tensor(index):
            index = index.int()
            return self.list[index] 
        else:
            assert isinstance(index, int) or index == int(index), f"Input {index} is not an int."
            assert int(index) in range(len(self.list)), f"Input {index} is out of range."
            return self.list[int(index)]

class Expand(CallableList):
    def __init__(self, start_index, start_p, end_p, max_n=1024):
        expand = [10,10] #start off with any values, doesn't matter
        finished_expanding = False
        for n in range(2, max_n+1):
            if n < start_index:
                a_n = (1-start_p) * expand[n-1] + start_p * expand[n-1]*n/(n-1)
                expand.append(a_n)
            else:
                a_n1 = 2*expand[n-1] - expand[n-2]
                a_n2 = (1-end_p) * expand[n-1] + end_p * expand[n-1]*n/(n-1)
                if a_n1 <= a_n2:
                    a_n = a_n1
                else:
                    a_n = a_n2
                    if not finished_expanding:
                        print(f"Expanded from {start_index} to {n}")
                        finished_expanding = True
                expand.append(a_n)
        super().__init__(expand)

class reduce_dimensions_activation(object):
    def __init__(self, a, max_n = 1024, epsilon=0.1):
        self.epsilon = epsilon
        if isinstance(a, list):
            a = CallableList(a)
        else:
            assert callable(a), "a is not a list or function"

        tolerance = 0.001
        for n in range(2, max_n + 1):
            assert a(n-1) <= a(n) <= a(n-1)*n/(n-1), f"a({n}) is not between a({n-1}) and {n}/{n-1} * a({n-1})."
            if n != 2:
                assert a(n) - a(n-1) <= a(n-1) - a(n-2) + tolerance, f"Difference between a({n}) and a({n-1}) is greater than the previous difference."

        self.a = a

    def __call__(self, t):        
        remaining_mask = torch.ones(t.shape).to(device)
        while True:
            n = torch.sum(remaining_mask, dim=-1)
            n_or_2 = torch.maximum(n, 2*torch.ones(n.shape).to(device))
            bound_constant = 1 - self.a(n_or_2-1)/self.a(n_or_2)
            new_remaining = 1*(torch.abs(t)*remaining_mask > torch.unsqueeze(torch.sum(torch.abs(t)*remaining_mask, dim=-1) * bound_constant, dim=-1))
            finished_mask = torch.logical_or(torch.eq(remaining_mask, new_remaining), torch.unsqueeze(torch.eq(n, torch.ones(n.shape).to(device)), dim=-1)) #finished if, for each set of activations, either no updates this loop or n = 1
            if torch.sum(~finished_mask) == 0:
                break
            remaining_mask = new_remaining

        k = torch.sum(remaining_mask, dim=-1)
        k_or_plus_1_or_2 = torch.maximum(torch.unsqueeze(k, dim=-1) + 1-remaining_mask, 2*torch.ones(t.shape).to(device))
        bound_constant = 1 - self.a(k_or_plus_1_or_2-1)/self.a(k_or_plus_1_or_2)
        bound = (torch.unsqueeze(torch.sum(torch.abs(t)*remaining_mask, dim=-1), dim=-1) + torch.abs(t) * (1 - remaining_mask)) * bound_constant
        return k, suppress_lower_activations(t, bound, epsilon=self.epsilon, inclusive=False, mode="absolute")


def suppress_lower_activations(t, bound, epsilon, inclusive=True, mode="absolute"):
    if torch.is_tensor(bound) and bound.numel() != 1:
        while bound.dim() < t.dim():
            bound = torch.unsqueeze(bound, -1)
    above_mask = (torch.abs(t) >= bound) if inclusive else (torch.abs(t) > bound)
    above_only = t * above_mask
    below_only = t * (~above_mask)
    if mode == "absolute":
        return above_only + epsilon/bound * below_only
    elif mode == "relative":
        return above_only + epsilon * below_only

def smoothed_piecewise(input, functions, transitions):
    assert len(functions) == len(transitions) + 1, "Incorrect number of transitions for number of functions given."
    for i in range(len(transitions)-1):
        assert transitions[i]["x"] < transitions[i+1]["x"], "Transition list not sorted by x-value in ascending order."
    sig = torch.nn.Sigmoid()
    sum = functions[0](input) #first add in the initial function
    for i, t in enumerate(transitions): #then at each transition we will subtract off the previous function and add on the next function
        g = functions[i]
        h = functions[i+1]
        if "focus" in t:
            if t["focus"] == "right":
                t["x"] = t["x"] - t["delta"]
                n = torch.log(abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"]))/t["epsilon"] - 1)/t["delta"]
            else:
                assert t["focus"] == "left", "Unrecognized focus for a transition (must be either right or left)."
                t["x"] = t["x"] + t["delta"]
                n = torch.log(abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))/t["epsilon"] - 1)/t["delta"]
        else:
            left_and_right = torch.stack((abs(g(t["x"]+t["delta"])-h(t["x"]+t["delta"])), abs(g(t["x"]-t["delta"])-h(t["x"]-t["delta"]))))
            n = torch.log(torch.max(left_and_right, dim=0).values/t["epsilon"] - 1)/t["delta"]
        sum += sig(n*(input-t["x"])) * h(input) - sig(n*(input-t["x"])) * g(input)
    return sum