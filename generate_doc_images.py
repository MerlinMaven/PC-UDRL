
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_hindsight_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Timeline
    T = 100
    t_current = 30
    
    # Draw trajectory line (curve)
    x = np.linspace(0, T, 100)
    y = np.sin(x/20) + np.random.normal(0, 0.05, 100) + 2
    
    ax.plot(x, y, color='gray', alpha=0.5, linestyle='--', label='Full Trajectory')
    
    # Highlight hindsight segment
    x_seg = x[t_current:]
    y_seg = y[t_current:]
    ax.plot(x_seg, y_seg, color='#2ca02c', linewidth=3, label='Hindsight Relabeled Segment')
    
    # Points
    ax.scatter([t_current], [y[t_current]], color='black', zorder=5)
    ax.text(t_current, y[t_current]+0.2, 'Current State $s_t$', ha='center')
    
    ax.scatter([T], [y[-1]], color='red', zorder=5)
    ax.text(T, y[-1]+0.2, 'Terminal State $s_T$', ha='center')
    
    # Annotations
    ax.annotate(
        '', xy=(T, 1), xytext=(t_current, 1),
        arrowprops=dict(arrowstyle='<->', color='blue', lw=2)
    )
    ax.text((T+t_current)/2, 0.7, 'Horizon $h = T - t$', ha='center', color='blue', fontsize=12)
    
    ax.annotate(
        '', xy=(T, y[-1]), xytext=(t_current, y[t_current]),
        arrowprops=dict(arrowstyle='-', color='orange', lw=2, connectionstyle="arc3,rad=-0.2")
    )
    ax.text((T+t_current)/2, 3.2, r'Return $r = \sum_{k=t}^T r_k$', ha='center', color='#d35400', fontsize=12)

    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title("Hindsight Relabeling: Learning from the Future", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('assets/hindsight_relabeling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated hindsight_relabeling.png")

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define boxes
    def draw_box(x, y, w, h, text, color, subtext=None):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", ec="black", fc=color, alpha=0.3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.1 if subtext else 0), text, ha='center', va='center', fontsize=12, fontweight='bold')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.15, subtext, ha='center', va='center', fontsize=9, style='italic')
        return x+w, y+h/2 # Return connection point (right)

    # Coordinates
    y_main = 0.5
    
    # PESSIMIST
    pess_x, pess_y = 0.35, 0.6
    draw_box(pess_x, pess_y, 0.3, 0.2, "PESSIMIST\n(Safety Shield)", "#e74c3c", "Cmd Projection")
    
    # USER
    user_x, user_y = 0.05, 0.6
    draw_box(user_x, user_y, 0.2, 0.2, "USER", "#95a5a6", "Target Return")
    
    # AGENT
    agent_x, agent_y = 0.35, 0.2
    draw_box(agent_x, agent_y, 0.3, 0.2, "AGENT\n(Policy)", "#3498db", "Action Selection")
    
    # ENV
    env_x, env_y = 0.75, 0.4
    draw_box(env_x, env_y, 0.2, 0.2, "ENVIRONMENT", "#2ecc71", "Simulatior")

    # Arrows
    def draw_arrow(x1, y1, x2, y2, label=None, color="black", bent=False):
        style = "Simple,tail_width=0.5,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color=color)
        if bent:
            conn = "arc3,rad=0.2"
            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=conn, **kw)
        else:
            arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), **kw)
        ax.add_patch(arrow)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.05, label, ha='center', fontsize=10, backgroundcolor='white')

    # User -> Pessimist
    draw_arrow(user_x+0.2, user_y+0.1, pess_x, pess_y+0.1, "$r_{cmd}$ (Unsafe)")
    
    # Pessimist -> Agent
    draw_arrow(pess_x+0.15, pess_y, agent_x+0.15, agent_y+0.2, "$r_{safe}$ (Clamped)")
    
    # Agent -> Env
    draw_arrow(agent_x+0.3, agent_y+0.1, env_x, env_y+0.1, "Action $a$", bent=True)
    
    # Env -> Agent/Pessimist (State)
    draw_arrow(env_x, env_y+0.1, pess_x+0.3, pess_y+0.1, "", bent=True, color="gray") # To Pessimist
    draw_arrow(env_x, env_y+0.1, agent_x+0.3, agent_y+0.1, "State $s$", bent=True, color="gray") # To Agent
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("PC-UDRL Inference Architecture", fontsize=14)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("PC-UDRL Inference Architecture", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('assets/system_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated system_architecture.png")

def create_dataset_pie_chart():
    fig, ax = plt.subplots(figsize=(6, 4))
    
    labels = ['Expert (50%)', 'Medium-Replay (30%)', 'Random (20%)']
    sizes = [50, 30, 20]
    colors = ['#a8dadc', '#f1faee', '#e63946'] # Modern Palette (Teal, White-ish, Red)
    # colors = ['#457b9d', '#a8dadc', '#e63946'] 
    colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Flat UI (Green, Yellow, Red)

    wedges, texts, autotexts = ax.pie(sizes, colors=colors, startangle=90, 
                                      wedgeprops=dict(width=0.5), autopct='%1.0f%%', pctdistance=0.75)
    
    # Legend
    ax.legend(wedges, labels, title="Sources", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=10, weight="bold")
    
    ax.set_title("LunarLander Dataset Composition", fontsize=14, pad=20)

    ax.set_title("LunarLander Dataset Composition", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('assets/dataset_pie.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated dataset_pie.png")

if __name__ == "__main__":
    create_hindsight_diagram()
    create_architecture_diagram()
    create_dataset_pie_chart()
