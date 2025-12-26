
import os
import sys
import torch
from torchviz import make_dot
import graphviz

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pc_udrl.agents.udrl_agent import UDRLAgent
from src.pc_udrl.pessimists.quantile import QuantileRegressor

def visualize_models():
    print("Generating model graphs...")
    output_dir = os.path.join("outputs", "graphs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Visualize UDRLAgent
    print("Visualizing UDRLAgent...")
    # GridWorld params: state_dim=2, action_dim=4, discrete=True
    agent = UDRLAgent(state_dim=2, action_dim=4, discrete=True, hidden_dim=256)
    
    # Dummy inputs
    dummy_state = torch.randn(1, 2)
    dummy_horizon = torch.randn(1)
    dummy_return = torch.randn(1)
    
    # Forward pass
    output_agent = agent(dummy_state, dummy_horizon, dummy_return)
    
    # Generate dot
    # params=dict(agent.named_parameters()) adds parameter names to the graph
    dot_agent = make_dot(output_agent, params=dict(agent.named_parameters()), show_attrs=True, show_saved=True)
    dot_agent.format = 'png'
    output_path_agent = os.path.join(output_dir, "udrl_agent_graph")
    dot_agent.render(output_path_agent)
    print(f"saved to {output_path_agent}.png")

    # 2. Visualize QuantileRegressor
    print("Visualizing QuantileRegressor...")
    pessimist = QuantileRegressor(state_dim=2, hidden_dim=256, q=0.9)
    
    # Forward pass
    output_pessimist = pessimist(dummy_state)
    
    dot_pessimist = make_dot(output_pessimist, params=dict(pessimist.named_parameters()), show_attrs=True, show_saved=True)
    dot_pessimist.format = 'png'
    output_path_pessimist = os.path.join(output_dir, "quantile_pessimist_graph")
    dot_pessimist.render(output_path_pessimist)
    print(f"saved to {output_path_pessimist}.png")

    print("Done.")

if __name__ == "__main__":
    try:
        visualize_models()
    except graphviz.backend.ExecutableNotFound:
        print("\nERROR: Graphviz executable not found.")
        print("Please install Graphviz from https://graphviz.org/download/ and add it to your PATH.")
        print("Alternatively, on Windows with chocolatey: choco install graphviz")
        print("Or with conda: conda install python-graphviz")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
