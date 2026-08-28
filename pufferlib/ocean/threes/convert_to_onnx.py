import torch
import torch.nn as nn

from pufferlib.ocean.torch import Threes, Recurrent

# Mock environment to avoid C binding / NumPy issues
class MockEnv:
    class single_observation_space:
        shape = (21,)
    class single_action_space:
        n = 4  # 4 actions: up, down, left, right

class RecurrentONNXWrapper(nn.Module):
    """Wrapper to make the Recurrent model ONNX-exportable with flat inputs/outputs."""
    def __init__(self, recurrent_model):
        super().__init__()
        self.model = recurrent_model
        self.hidden_size = recurrent_model.hidden_size

    def forward(self, observations, lstm_h, lstm_c):
        # Create state dict expected by the model
        state = {
            'lstm_h': lstm_h,
            'lstm_c': lstm_c
        }

        # Run forward_eval (inference mode)
        logits, value = self.model.forward_eval(observations, state)

        # Return outputs including new LSTM state
        return logits, value, state['lstm_h'], state['lstm_c']

def main():
    # Create base policy (must match training config)
    base_policy = Threes(MockEnv(), hidden_size=1024, embed_dim=8)

    # Wrap with Recurrent (LSTM) - must match training config
    recurrent_model = Recurrent(MockEnv(), base_policy, input_size=1024, hidden_size=1024)

    # Load checkpoint
    checkpoint = torch.load('../../../experiments/puffer_threes_nnm6a1s9.pt', map_location='cpu')

    # Load all weights (no filtering needed - load directly)
    recurrent_model.load_state_dict(checkpoint)
    recurrent_model.eval()

    # Test inference before export
    print("Testing PyTorch model...")
    obs = torch.tensor([[0, 1, 2, 3, 3, 0, 1, 2, 0, 0, 3, 0, 1, 2, 0, 0, 7, 4, 4, 4, 0]], dtype=torch.long)
    h = torch.zeros(1, 1024)
    c = torch.zeros(1, 1024)
    state = {'lstm_h': h, 'lstm_c': c}

    with torch.no_grad():
        logits, value = recurrent_model.forward_eval(obs, state)

    probs = torch.softmax(logits, dim=-1)
    print(f"  Logits: {logits[0].tolist()}")
    print(f"  Probs: {probs[0].tolist()}")
    print(f"  Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][torch.argmax(logits).item()]}")

    # Create ONNX wrapper
    onnx_model = RecurrentONNXWrapper(recurrent_model)
    onnx_model.eval()

    # Dummy inputs for ONNX export
    dummy_obs = torch.zeros(1, 21, dtype=torch.long)
    dummy_h = torch.zeros(1, 1024)
    dummy_c = torch.zeros(1, 1024)

    print("\nExporting to ONNX...")
    # Use legacy exporter for better compatibility
    torch.onnx.export(
        onnx_model,
        (dummy_obs, dummy_h, dummy_c),
        'onnx_model.onnx',
        input_names=['observations', 'lstm_h', 'lstm_c'],
        output_names=['logits', 'value', 'lstm_h_out', 'lstm_c_out'],
        dynamic_axes={
            'observations': {0: 'batch'},
            'lstm_h': {0: 'batch'},
            'lstm_c': {0: 'batch'},
            'logits': {0: 'batch'},
            'value': {0: 'batch'},
            'lstm_h_out': {0: 'batch'},
            'lstm_c_out': {0: 'batch'},
        },
        opset_version=13,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        dynamo=False  # Use legacy TorchScript exporter
    )

    print("ONNX export complete: onnx_model.onnx")
    print("\nCopy to threes_web/model/:")
    print("  cp onnx_model.onnx* /path/to/threes_web/model/")

if __name__ == '__main__':
    main()
